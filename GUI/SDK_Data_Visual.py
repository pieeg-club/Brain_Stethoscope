"""
pieeg - Real-Time 16-Channel EEG + Heatmap + FFT
=====================================================
Requirements:
    pip install bleak matplotlib numpy scipy

Usage:
    python pieeg.py --demo
    python pieeg.py
    python pieeg.py --window 6 --save data.csv
    python pieeg.py --scan
    python pieeg.py --notch 60
"""

import asyncio
import argparse
import csv
import sys
import time
import threading
import collections

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, lfilter
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon

SERVICE_UUID        = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_CHAR_UUID      = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME         = "bioron_16"
PACKET_SIZE         = 51
HEADER_BYTE         = 0xA0
FOOTER_BYTE         = 0xC0
NUM_CHANNELS        = 16
BLE_SCAN_TIMEOUT    = 30.0
BLE_CONNECT_TIMEOUT = 20.0
DESIRED_MTU         = 100

EEG_BANDS = [
    (0.5,  4,  "#1a2a3a", "delta"),
    (4,    8,  "#1a2e1a", "theta"),
    (8,   13,  "#2e1a1a", "alpha"),
    (13,  30,  "#2e2200", "beta"),
    (30,  70,  "#1e1530", "gamma"),
]

EEG_NAMES = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3',
             'C3', 'Cz', 'C4','T4','P3','Pz','P4','Oz']
EEG_XY = np.array([
    (-0.27,  0.85), ( 0.27,  0.85),
    (-0.72,  0.45), (-0.38,  0.50), ( 0.00,  0.50), ( 0.38,  0.50), ( 0.72,  0.45),
    (-0.85,  0.00), (-0.45,  0.00), ( 0.00,  0.00), ( 0.45,  0.00), ( 0.85,  0.00),
    (-0.38, -0.50), ( 0.00, -0.50), ( 0.38, -0.50),
    ( 0.00, -0.85),
])

COLORS = [
    "#e63946","#f4a261","#2a9d8f","#457b9d",
    "#a8dadc","#e9c46a","#7fc97f","#8ecae6",
    "#ff6b6b","#ffd166","#06d6a0","#118ab2",
    "#fb8500","#8338ec","#3a86ff","#ff006e",
]
CHANNEL_LABELS = [f"CH{i+1}" for i in range(NUM_CHANNELS)]

EEG_CMAP = mcolors.LinearSegmentedColormap.from_list("eeg", [
    (0.00, "#3a86ff"), (0.25, "#06d6a0"),
    (0.50, "#ffd166"), (0.75, "#fb8500"),
    (1.00, "#e63946"),
])

HEATMAP_RES = 160

def _build_idw_grid():
    xs = np.linspace(-1.1, 1.1, HEATMAP_RES)
    ys = np.linspace(-1.1, 1.1, HEATMAP_RES)
    gx, gy = np.meshgrid(xs, ys)
    mask = (gx**2 + gy**2) <= 1.0
    w = np.zeros((HEATMAP_RES, HEATMAP_RES, NUM_CHANNELS))
    for i, (ex, ey) in enumerate(EEG_XY):
        d2 = (gx - ex)**2 + (gy - ey)**2
        w[:, :, i] = 1.0 / (d2 + 1e-6)
    w /= w.sum(axis=2, keepdims=True)
    return mask, w

_CIRCLE_MASK, _IDW_W = _build_idw_grid()

def interpolate_heatmap(vals_norm):
    img = (_IDW_W * np.array(vals_norm)[np.newaxis, np.newaxis, :]).sum(axis=2)
    img[~_CIRCLE_MASK] = np.nan
    return img

def highpass_filter(data, cutoff=0.5, fs=250.0, order=4):
    if len(data) < order * 3 + 1:
        return data
    sos = butter(order, cutoff, btype="high", fs=fs, output="sos")
    return sosfilt(sos, data)

def notch_filter(data, freq=50.0, fs=250.0, q=30.0):
    if len(data) < 10:
        return data
    b, a = iirnotch(freq, q, fs)
    return lfilter(b, a, data)

def apply_filters(data, fs, notch_freq):
    out = highpass_filter(data, cutoff=0.5, fs=fs)
    out = notch_filter(out, freq=notch_freq, fs=fs)
    return out

def parse_int24_be(b0, b1, b2):
    val = (b0 << 16) | (b1 << 8) | b2
    return val - 0x1000000 if val & 0x800000 else val

def parse_packet(data: bytes):
    if len(data) != PACKET_SIZE or data[0] != HEADER_BYTE or data[50] != FOOTER_BYTE:
        return None
    channels = []
    for i in range(8):
        b = 2 + i * 3
        channels.append(parse_int24_be(data[b], data[b+1], data[b+2]))
    for i in range(8):
        b = 26 + i * 3
        channels.append(parse_int24_be(data[b], data[b+1], data[b+2]))
    return {"counter": data[1], "channels": channels}

SPS_HINT = 250

class DataStore:
    def __init__(self, window_sec: float, sps_hint: int = SPS_HINT):
        self.window_sec     = window_sec
        self.sps_hint       = sps_hint
        self.window_samples = int(window_sec * sps_hint)
        cap = max(self.window_samples * 2, 2048)
        self.buffers        = [collections.deque(maxlen=cap) for _ in range(NUM_CHANNELS)]
        self.lock           = threading.Lock()
        self.sample_count   = 0
        self.dropped_count  = 0
        self.last_counter   = None
        self.start_time     = None
        self.csv_writer     = None
        self.csv_file       = None

        # KEY FIX: pre-fill every channel with zeros equal to window_samples.
        # snapshot_window() always slices the last window_samples entries,
        # so these zeros fill the left side until real data arrives.
        for _ in range(self.window_samples):
            for ch in range(NUM_CHANNELS):
                self.buffers[ch].append(0)

    def open_csv(self, path: str):
        self.csv_file   = open(path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "counter"] + CHANNEL_LABELS)

    def close_csv(self):
        if self.csv_file:
            self.csv_file.close()

    def push(self, channels, counter, ts):
        with self.lock:
            if self.start_time is None:
                self.start_time = ts
            if self.last_counter is not None:
                exp = (self.last_counter + 1) & 0xFF
                if counter != exp:
                    self.dropped_count += (counter - self.last_counter - 1) & 0xFF
            self.last_counter  = counter
            self.sample_count += 1
            for ch, val in enumerate(channels):
                self.buffers[ch].append(val)
            if self.csv_writer:
                self.csv_writer.writerow([f"{ts:.6f}", counter] + channels)

    def snapshot_window(self):
        """
        Returns (t, [ch0..ch15]) where t spans exactly [-window_sec, 0]
        and each channel has exactly window_samples points.
        Uses sample index for time — never relies on wall-clock timestamps.
        """
        n = self.window_samples
        sps = self.sps
        with self.lock:
            ch = [np.array(list(b)[-n:], dtype=float) for b in self.buffers]
        # Pad any channel that is still shorter than n (should not happen after prefill)
        for i in range(NUM_CHANNELS):
            if len(ch[i]) < n:
                ch[i] = np.concatenate([np.zeros(n - len(ch[i])), ch[i]])
        t = np.linspace(-self.window_sec, 0, n, endpoint=False)
        return t, ch

    def snapshot_full(self):
        with self.lock:
            return [np.array(b, dtype=float) for b in self.buffers]

    def latest(self):
        with self.lock:
            return [float(b[-1]) if b else 0.0 for b in self.buffers]

    @property
    def sps(self):
        if self.start_time and self.sample_count > 1:
            return self.sample_count / (time.time() - self.start_time)
        return float(self.sps_hint)


async def _scan_all_devices(timeout=10.0):
    from bleak import BleakScanner
    print(f"[BLE] Scanning all devices for {timeout:.0f}s ...")
    devices = await BleakScanner.discover(timeout=timeout, return_adv=True)
    if not devices:
        print("[BLE] No devices found.")
        return
    for addr, (dev, adv) in devices.items():
        uuids = ", ".join(str(u) for u in adv.service_uuids) or "-"
        print(f"  {addr}  {dev.name!r:20s}  {adv.rssi} dBm  {uuids}")

async def _find_device():
    from bleak import BleakScanner
    print("[BLE] Scanning by service UUID ...")
    dev = await BleakScanner.find_device_by_filter(
        lambda d, adv: SERVICE_UUID.lower() in
                       [str(u).lower() for u in adv.service_uuids],
        timeout=BLE_SCAN_TIMEOUT,
    )
    if dev:
        print(f"[BLE] Found by UUID: {dev.name!r}  {dev.address}")
        return dev
    print(f"[BLE] Trying name '{DEVICE_NAME}' ...")
    dev = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=BLE_SCAN_TIMEOUT)
    if dev:
        print(f"[BLE] Found by name: {dev.name!r}  {dev.address}")
        return dev
    print("[BLE] Not found. Listing all visible devices:")
    await _scan_all_devices(10.0)
    return None

def ble_thread_main(store, stop_event):
    asyncio.run(_ble_async(store, stop_event))

async def _ble_async(store, stop_event):
    from bleak import BleakClient
    device = await _find_device()
    if device is None:
        print("[ERR] Device not found.")
        stop_event.set()
        return

    def on_data(_h, data: bytearray):
        r = parse_packet(bytes(data))
        if r:
            store.push(r["channels"], r["counter"], time.time())

    print(f"[BLE] Connecting to {device.address} ...")
    try:
        async with BleakClient(device, timeout=BLE_CONNECT_TIMEOUT) as client:
            if hasattr(client, "request_mtu"):
                try:
                    await client.request_mtu(DESIRED_MTU)
                except Exception as e:
                    print(f"[BLE] MTU req failed (ok): {e}")
            if client.mtu_size < PACKET_SIZE + 3:
                print(f"[WARN] MTU {client.mtu_size} too small")
            print(f"[BLE] Connected  MTU={client.mtu_size}")
            await client.start_notify(DATA_CHAR_UUID, on_data)
            print("[BLE] Streaming ... close window to stop.")
            while not stop_event.is_set():
                await asyncio.sleep(0.1)
            await client.stop_notify(DATA_CHAR_UUID)
    except Exception as e:
        print(f"[ERR] BLE: {e}")
        stop_event.set()

def demo_thread_main(store, stop_event):
    freqs   = [1, 2, 4, 8, 10, 13, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    counter = 0
    t0      = time.time()
    while not stop_event.is_set():
        t  = time.time() - t0
        ch = [int(np.sin(2 * np.pi * f * t) * 500_000 + np.random.randn() * 8000)
              for f in freqs]
        store.push(ch, counter & 0xFF, time.time())
        counter += 1
        time.sleep(1 / 250)

FFT_WIN_SEC  = 2.0
FFT_MAX_FREQ = 70.0

def compute_fft(ch_data, sps):
    n = len(ch_data)
    if n < 32 or sps < 1:
        return None, None
    win  = np.hanning(n)
    sig  = (ch_data - ch_data.mean()) * win
    spec = np.abs(np.fft.rfft(sig)) * 2 / n
    freq = np.fft.rfftfreq(n, d=1.0 / sps)
    mask = freq <= FFT_MAX_FREQ
    return freq[mask], spec[mask]

BG = "#0d1117"

class FullPlotter:
    def __init__(self, store: DataStore, window_sec: float, notch_freq: float = 50.0):
        self.store      = store
        self.window_sec = window_sec
        self.notch_freq = notch_freq
        self._peak      = 1.0

        self.fig = plt.figure(figsize=(22, 11), facecolor=BG)
        gs_outer = self.fig.add_gridspec(
            1, 2, width_ratios=[2.2, 1],
            left=0.055, right=0.975,
            top=0.955, bottom=0.045,
            wspace=0.06,
        )
        gs_left  = gs_outer[0].subgridspec(NUM_CHANNELS, 1, hspace=0.0)
        self.axes_eeg = [self.fig.add_subplot(gs_left[i]) for i in range(NUM_CHANNELS)]
        gs_right = gs_outer[1].subgridspec(2, 1, hspace=0.38, height_ratios=[1.1, 1])
        self.ax_heat = self.fig.add_subplot(gs_right[0])
        self.ax_fft  = self.fig.add_subplot(gs_right[1])

        self._setup_waterfall()
        self._setup_heatmap()
        self._setup_fft()

        self.status = self.fig.text(
            0.99, 0.003, "", color="#444", fontsize=8, ha="right", va="bottom"
        )

    def _setup_waterfall(self):
        self.eeg_lines = []
        for idx, ax in enumerate(self.axes_eeg):
            ax.set_facecolor(BG)
            ax.tick_params(colors="#333", labelsize=5.5, length=2, pad=1)
            for sp in ax.spines.values():
                sp.set_color("#1e1e1e")
            ax.yaxis.set_major_locator(ticker.MaxNLocator(2, prune="both"))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda v, _: (f"{v/1e6:.1f}M" if abs(v) >= 1e6
                              else f"{v/1e3:.0f}k" if abs(v) >= 1e3
                              else f"{v:.0f}")
            ))
            # Set x-axis once here — never touch it again in _update
            ax.set_xlim(-self.window_sec, 0)
            c = COLORS[idx]
            line, = ax.plot([], [], color=c, linewidth=0.55, antialiased=True)
            self.eeg_lines.append(line)
            ax.text(-0.043, 0.5, CHANNEL_LABELS[idx],
                    transform=ax.transAxes, color=c, fontsize=6.5,
                    fontweight="bold", va="center", ha="right")
        self.axes_eeg[-1].set_xlabel("Time (s)", color="#555", fontsize=7.5)
        self.axes_eeg[-1].tick_params(axis="x", colors="#555", labelsize=7)
        self.fig.text(0.22, 0.968, "pieeg - 16-channel live stream",
                      color="white", fontsize=10, fontweight="bold",
                      ha="center", va="top")

    def _setup_heatmap(self):
        ax = self.ax_heat
        ax.set_facecolor(BG)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.25, 1.38)
        ax.axis("off")
        blank = np.full((HEATMAP_RES, HEATMAP_RES), np.nan)
        self.hmap_img = ax.imshow(
            blank, extent=[-1.1, 1.1, -1.1, 1.1], origin="lower",
            cmap=EEG_CMAP, vmin=0, vmax=1, interpolation="bilinear", zorder=1,
        )
        ax.add_patch(Circle((0, 0), 1.0, fill=False,
                            edgecolor="#555", linewidth=1.2, zorder=4))
        ax.annotate("", xy=(0, 1.18), xytext=(0, 1.0),
                    arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.0), zorder=4)
        for ex in (-1.0, 1.0):
            ax.add_patch(Polygon(
                [(ex, 0.1), (ex*1.12, 0.06), (ex*1.14, -0.06), (ex, -0.1)],
                fill=False, edgecolor="#555", linewidth=1.0, zorder=4))
        self._dots = []
        for i, (ex, ey) in enumerate(EEG_XY):
            dot, = ax.plot(ex, ey, 'o', markersize=8,
                           markerfacecolor="#1a1a1a", markeredgecolor="#666",
                           markeredgewidth=0.7, zorder=5)
            ax.text(ex, ey + 0.1, EEG_NAMES[i], color="#888", fontsize=5.5,
                    ha="center", va="bottom", zorder=6)
            self._dots.append(dot)
        cbar = self.fig.colorbar(self.hmap_img, ax=ax, orientation="horizontal",
                                 fraction=0.04, pad=0.01, aspect=28)
        cbar.ax.tick_params(colors="#555", labelsize=6.5)
        cbar.outline.set_edgecolor("#333")
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(["-peak", "0", "+peak"])
        ax.set_title("Topographic heatmap", color="#aaa", fontsize=8, pad=4)

    def _setup_fft(self):
        ax = self.ax_fft
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color("#222")
        ax.tick_params(colors="#555", labelsize=7)
        for f0, f1, col, name in EEG_BANDS:
            ax.axvspan(f0, f1, color=col, alpha=0.7, zorder=0)
            ax.text((f0+f1)/2, 0, name, color="#667", fontsize=6,
                    ha="center", va="bottom",
                    transform=ax.get_xaxis_transform(), zorder=1)
        self.fft_lines = []
        for idx in range(NUM_CHANNELS):
            line, = ax.plot([], [], color=COLORS[idx],
                            linewidth=0.55, alpha=0.75, antialiased=True)
            self.fft_lines.append(line)
        ax.set_xlim(0, FFT_MAX_FREQ)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frequency (Hz)", color="#666", fontsize=7.5)
        ax.set_ylabel("Amplitude", color="#666", fontsize=7.5)
        ax.set_title(
            f"FFT  0-{int(FFT_MAX_FREQ)} Hz  (all 16 ch)  |  notch {self.notch_freq:.0f} Hz",
            color="#aaa", fontsize=8, pad=4)
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.grid(which="major", color="#1e1e1e", linewidth=0.5)
        ax.grid(which="minor", color="#181818", linewidth=0.3)

    def _update(self, _frame):
        sps = self.store.sps

        # Waterfall — t always spans [-window_sec, 0], all channels same length
        t_win, ch_arrs = self.store.snapshot_window()
        for idx, (line, ax) in enumerate(zip(self.eeg_lines, self.axes_eeg)):
            raw = ch_arrs[idx]
            if len(raw) < 4:
                continue
            y = apply_filters(raw, fs=sps, notch_freq=self.notch_freq)
            line.set_data(t_win, y)
            ymin, ymax = y.min(), y.max()
            span = max(ymax - ymin, 1)
            pad  = span * 0.1
            ax.set_ylim(ymin - pad, ymax + pad)

        # Heatmap
        latest  = self.store.latest()
        max_abs = max(abs(v) for v in latest) or 1.0
        if max_abs > self._peak:
            self._peak = max_abs
        self._peak *= 0.9997
        vals_norm = [(v / self._peak + 1) / 2 for v in latest]
        self.hmap_img.set_data(interpolate_heatmap(vals_norm))
        for i, dot in enumerate(self._dots):
            dot.set_markerfacecolor(EEG_CMAP(vals_norm[i]))
            dot.set_markeredgecolor("white")

        # FFT
        ch_full  = self.store.snapshot_full()
        fft_n    = max(int(FFT_WIN_SEC * sps), 64)
        fft_ymax = 0.0
        for idx in range(NUM_CHANNELS):
            data = ch_full[idx]
            if len(data) < 32:
                self.fft_lines[idx].set_data([], [])
                continue
            seg = data[-min(len(data), fft_n):]
            seg = apply_filters(seg, fs=sps, notch_freq=self.notch_freq)
            freq, amp = compute_fft(seg, sps)
            if freq is None:
                self.fft_lines[idx].set_data([], [])
                continue
            self.fft_lines[idx].set_data(freq, amp)
            fft_ymax = max(fft_ymax, amp.max())
        if fft_ymax > 0:
            self.ax_fft.set_ylim(0, fft_ymax * 1.15)

        self.status.set_text(
            f"samples: {self.store.sample_count}   sps: {sps:.1f}   "
            f"dropped: {self.store.dropped_count}   peak: {self._peak/1e3:.1f}k   "
            f"hp: 0.5 Hz   notch: {self.notch_freq:.0f} Hz"
        )
        return self.eeg_lines + self.fft_lines + [self.hmap_img]

    def run(self, stop_event: threading.Event):
        self.anim = FuncAnimation(
            self.fig, self._update, interval=50,
            blit=False, cache_frame_data=False,
        )
        self.fig.canvas.mpl_connect("close_event", lambda _: stop_event.set())
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="pieeg - waterfall + heatmap + FFT")
    parser.add_argument("--window", type=float, default=4.0)
    parser.add_argument("--save",   metavar="FILE")
    parser.add_argument("--demo",   action="store_true")
    parser.add_argument("--scan",   action="store_true")
    parser.add_argument("--notch",  type=float, default=50.0)
    args = parser.parse_args()

    if args.scan:
        asyncio.run(_scan_all_devices(timeout=10.0))
        sys.exit(0)

    store      = DataStore(window_sec=args.window)
    stop_event = threading.Event()

    if args.save:
        store.open_csv(args.save)
        print(f"[CSV] Logging to {args.save}")

    if args.demo:
        print("[DEMO] Synthetic 16-channel data ...")
        worker = threading.Thread(target=demo_thread_main,
                                  args=(store, stop_event), daemon=True)
    else:
        worker = threading.Thread(target=ble_thread_main,
                                  args=(store, stop_event), daemon=True)

    worker.start()
    FullPlotter(store, window_sec=args.window, notch_freq=args.notch).run(stop_event)
    stop_event.set()
    store.close_csv()
    print(f"\n[DONE] {store.sample_count} samples | "
          f"{store.sps:.1f} sps | {store.dropped_count} dropped")

if __name__ == "__main__":
    main()
