#!/usr/bin/env python

# (Hiroto, 2025/11/21) OGA16_ver1.0.0: Read baseband data --> (No 2nd beamforming) --> IncDD
# (Hiroto, 2025/11/27) OGA16_ver1.1.0: [Revise] loadFullbitmap() >> memmep()
# (Hiroto, 2025/12/02) OGA16_ver2.0.0: [Modify] Add dispersed pulse waterfall
# (Hiroto, 2025/12/05) OGA16_ver3.0.0: [Modify] Read single beam >> Improve the memory effeciency
# (Hiroto, 2025/12/10) OGA16_ver3.0.1: [Debug] p1<0 -> p1 = L + p1
# (Hiroto, 2026/01/01) OGA16_ver3.1.0: [Revise] Ring buffer p1<0 -> p1 = L + p1 | if (p2 > packMax):p2 = packMax | if (no_bitmap): bitmap=np.array(nPack)
# (Hiroto, 2026/01/06) OGA16_ver3.1.1: [Debug] The presence of NaN values corrupted the SNR calculation results: np.mean() & np.std() >> np.nanmean() & np.nanstd()
# (Hiroto, 2026/01/07) OGA16_ver4.0.0: [Modify] freq_ref=400MHz >> 300MHz
# (Hiroto, 2026/03/07) OGA16_ver4.0.1: [Debug] nBlock=40 >> 96
# (Hiroto, 2026/04/20) bf16_ver5.0.0: [Modify] Genelization: Define STATION_CONFIG and build_station_params() | Add options (--nSum, --nBin, --secWin, --freq_ref, --beam, --nblock, --station) 
# (Hiroto, 2026/04/20) bf16_ver5.0.1: [Debug] Add spec = spec[:nTime * args.nSum] for inconsistent data size
# (Hiroto, 2026/04/29) bf16_ver5.1.0: [Revise] Add ep for each channels along with ep0 at 400MHz

####  import necessary Modules ##########

import sys, os.path, time, re
import shlex
import argparse
import subprocess
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from glob import glob
from packet_func import *
#print(sys.path)
#sys.path.append('/data/kylin/beamform')
# sys.path.append('/data/hmasaoka/analysis/ddpacktrigger/beamform')
# sys.path.append('../beamform')
# sys.path.append('./beamform')
# import beamform as bf
#help(bf)
import multiprocessing as mp

from astropy.stats import sigma_clip
import pandas as pd
from astropy.time import Time

cmd = " ".join(shlex.quote(arg) for arg in sys.argv)
print(f"Command: {cmd}")
print("===============")

# ============================================================
# Logging Utility
# ============================================================
def print_log(level: str, msg: str):
    """Unified logging with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level.upper():7}] {msg}")

# ============================================================
# Functions
# ============================================================
def load_bin_files(idir):
    if not os.path.isdir(idir):
        print(f"[WARN] Directory not found: {idir}")
        return []

    # Get *.bin
    files = glob(os.path.join(idir, "*.bin"))
    if not files:
        print(f"[INFO] No .bin files found in {idir}")
        return []
    
    # Sort files based on o<number>.bin
    def extract_order(f):
        m = re.search(r'o(\d+)\.bin$', os.path.basename(f))
        return int(m.group(1)) if m else -1

    files = sorted(files, key=extract_order)
    nFile = len(files)

    print(f"[INFO] Found {nFile} .bin files in {idir}")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    return files

# ============================================================
# Data Reader
# ============================================================

def loadBatch(fh, pack0, npack, bpp, ppf, beam, order_off=0, hdlen=64, bitwidth=4, hdver=1, meta=0, unswap=False):
    '''
    load npack blocks of binary data from the open filehandle
    starting from pack0. each block is of pack_len bytes.
    input:
        pack0: starting packet index (starting from 0)
        npack: number of packets to load
        bpp: bytes per packet (paylod only)
    optional:
        order_off: packet order offset (buggy, keep =0 for now)
        hdlen: header length in bytes
        bitwidth: data format keyword
        hdver: header format keyword (1,2)
        meta: ring buffer or file metadata length in bytes

    output:
        (masked array)
        data0, shape=(npack, bpp)
        clock, shape=(npack,)
        order, shape=(npack,)
    '''

    pack_len = hdlen + bpp
    b0 = pack0 * pack_len + meta
    fh.seek(b0)

    if (bitwidth==4):
        nsamp = bpp
        if (ppf == 2):      # 4bit, bf16, bf64 modes
            grp = 2; nGroup = 256
            nsamp = nGroup*grp # single beam size; nAnt=1
    elif (bitwidth==16):
        nsamp = bpp//4

    data0 = np.ma.array(np.zeros((npack, nsamp), dtype=np.complex64), mask=False)
    clock = np.ma.array(np.zeros(npack, dtype=np.int64), mask=False)
    order = np.ma.array(np.zeros(npack, dtype=int), mask=False)
    for i in range(npack):
        buf = fh.read(pack_len)
        tmp = packetUnpack(buf, bpp, order_off=order_off, hdlen=hdlen, bitwidth=bitwidth, hdver=hdver, unswap=unswap) # tmp=8192byte + a
        if (tmp is None):
            data0.mask[i] = True
            clock.mask[i] = True
            order.mask[i] = True
        else:
            clk, pko, spec = tmp
            antSpec = spec.reshape((nGroup, scfg["nAnt"], grp)).transpose((1, 0, 2)).reshape((scfg["nAnt"], nsamp)) # single beam size; nAnt=1 | beamSpec.shape=(nAnt, nGroup*grp)
            beamSpec = antSpec[beam] # single beam size; nAnt=1 | beamSpec.shape=(nGroup*grp,)
            data0[i] = beamSpec # single beam size; nAnt=1 | beamSpec.shape=(nGroup*grp,)
            clock[i] = clk
            order[i] = pko

    return data0, clock, order

def formSpec(data0, clock, order, ppf, nAnt=16, grp=2, nChan=1024, bitmap=None, verbose=0):
    '''
    form spectra from the packet series
    input:
        data0, clock, order: as output of loadBatch
                data0.shap = (npack, bpp)
        ppf: packet per frame (e.g. 2, 4 or 8)
    optional:
        grp: number of consecutive bytes (channels) from the same antenna input 
        bitmap: if provided, should be a boolean array of shape (npack,)
                True for good packet, False for bad packet
                also implies the packets have been ordered, contiguous,
                although some packets may be invalid

                on the other hand, if the bitmap is None,
                it assumes all packets are valid, but maybe not contiguous
    
    output:
        data_tick, shape=(ntick,)
        antSpec, shape=(ntick,nAnt,nChan)   where nChan=bpp*ppf/nAnt e.g. 8192*2/16=1024
    '''
    npack, bpp = data0.shape    # e.g. 1000, 8192

    # pre-defined format, ignore input
    if (ppf == 2):      # 4bit, bf16, bf64 modes
        grp = 2; nGroup = 256
        nsamp = nGroup*grp # single beam size; nAnt=1
    elif (ppf == 8):    # 16bit mode
        grp = 1

    ## rearranging packets into frames, taking into account lost packets
    #clock  = np.array(clock)   # already an array

    # raw tick
    tick = clock - order        # packet of the same frame should have the same tick
                                        # i.e. tick is the frame id, delta-tick = ppf

    if (bitmap is None):
        if (verbose>1):
            print('in formSpec: bitmap is None. generate ad hoc one.')
        # new tick
        tick3 = tick - tick[0]              # tick3 starts from 0, increase by ppf (if no packet lost)
        tick3 = tick3 // ppf                # tick3 increment normalized to 1
        ntick = tick3.max() + 1             # e.g. ntick=500 for npack=1000
        ntick = tick3[-1] - tick3[0] + 1
        valid = np.ones(npack, dtype=bool)
        tick0 = tick[0]
    else:
        ntick = npack // ppf
        tmp = np.arange(npack)
        #tmp -= tmp%ppf
        tmp -= order
        tick3 = (tmp//ppf).astype(int)
        valid = bitmap
        if (verbose>1):
            print('in formSpec: npack/nvalid', npack, np.count_nonzero(valid))
        tick0 = -1
        ii = 0
        while (tick0 < 0 and ii<npack):
            if (valid[ii]):
                tick0 = tick[ii] - ii
            else:
                ii += 1
        if (tick0 < 0):
            print('error finding valid tick')
            return None
            
    if (verbose>0):
        print('ntick, ppf, bpp:', ntick, ppf, bpp)
        #print('tick3 min,max:', tick3.min(), tick3.max())
    data  = np.ma.array(np.zeros((ntick, ppf, nsamp), dtype=np.complex64), mask=False)
    # data  = np.ma.array(np.zeros((ntick, ppf, nsamp), dtype=np.complex64), mask=True)
    # data  = np.ma.array(np.zeros((ntick, ppf, bpp), dtype=np.complex64), mask=True)
    data_tick = np.arange(ntick) + tick0
    for i in range(npack):
        if (valid[i]):
            if (order.mask[i] == False):
                it = int(tick3[i])
                io = int(order[i])
                #print('debug:', i, it, io)
                if (io >= ppf):
                    sys.exit('incorrect ppf. abort.')
                if (it >= 0):
                    data[it, io] = data0[i]
                    data[it, io].mask = False

    pack_flag = data.mask.all(axis=2)
    empty_frame = data.mask.all(axis=(1,2))
    fault_frame = data.mask.any(axis=(1,2))
    empty_pack  = data.mask.all(axis=(0,2))

    if (verbose>1): # debug info
        print('ntick:', ntick)
        print('nempty:', np.count_nonzero(empty_frame)) # frames that are totally empty
        print('nfault:', np.count_nonzero(fault_frame)) # frames that are partially empty

    if (ppf==2 or ppf==1):      # 4bit version
        antSpec = data.reshape((ntick,nChan))
        # antSpec = data.reshape((ntick,-1,nAnt,grp)).transpose((0,2,1,3)).reshape((ntick,nAnt,-1))
    elif (ppf==8):    # 16bit version
        evntmp = data[:,0:4].reshape((ntick,-1,nAnt)).transpose((0,2,1))
        oddtmp = data[:,4:8].reshape((ntick,-1,nAnt)).transpose((0,2,1))
        antSpec = np.ma.array(np.zeros((ntick,nAnt,nChan),dtype=complex), mask=True)
        antSpec[:,:,0::2] = evntmp
        antSpec[:,:,1::2] = oddtmp

    return data_tick, antSpec


def loadSpec(fh, pack0, npack, beam, bpp=8192, ppf=2, order_off=0, nAnt=16, grp=2, hdlen=64, bitmap=None, nBlock=0, verbose=0, bitwidth=4, hdver=1, meta=0, blocklen=128000, unswap=False):
    '''
    a wrapper of loadBatch + formSpec
    input:
        fh: open binary file handle
        pack0: starting packet index (starting from 0)
        npack: number packets to load
        bpp: bytes per packet (payload length, e.g. 8192)
        ppf: packet per frame (e.g. 2)

    optional:
        nAnt: number of antenna inputs in the packet
        grp: number of consecutive channels (bytes) of the same antenna
        hdlen: header length in bytes
        order_off: (buggy) shift the packet order around, keep =0 for now
        bitmap: boolean array, shape=(npack,); True for a good packet, False for bad
        nBlock: int; number of Block in the open file (fh)
                if bitmap is None and nBlock>0, the bitmap is automatically loaded
        verbose: int; pass to formSpec
        bitwidth: size of the I/Q data (4 or 16)
                (automatically determines the grp and ppf)
        hdver: header version (1, 2)
        meta: ring buffer or file metadata length in bytes

    output:
        data_tick, shape=(ntick,)
        antSpec, shape=(ntick, nAnt, nChan)
    '''
    #print('debug: meta', meta)
    if (meta == 64):    # assuming a valid file header
        # override default nBlock and blocklen
        mdict = metaRead(fh)
        blocklen = mdict['packet_number']
        nBlock = mdict['block_number']
        # also override hdver
        hdver = 2
        if (verbose):
            print('nBlock', nBlock, 'blocklen', blocklen)

    if (bitmap is None and nBlock>0):
        BM = loadFullbitmap(fh, nBlock, blocklen=blocklen, meta=meta)
        bitmap = BM[pack0:pack0+npack]

    if (bitmap is not None):
        nvalid = np.count_nonzero(bitmap)
        fvalid = float(nvalid/npack)
        if (verbose>0):
            print('pack0, npack, nvalid:', pack0, npack, nvalid)
        if (nvalid == 0):
            print('no valid packet!')
            return None
    else:
        if (verbose>0):
            print('pack0, npack:', pack0, npack, 'bitmap N/A')

    if (bitwidth==16):
        grp = 1
        ppf = 8

    data0, clock, order = loadBatch(fh, pack0, npack, bpp, ppf, beam, order_off=order_off, hdlen=hdlen, bitwidth=bitwidth, hdver=hdver, meta=meta, unswap=unswap)
    tmp = formSpec(data0, clock, order, ppf, nAnt=nAnt, grp=grp, bitmap=bitmap, verbose=verbose)

    if (tmp is not None):
        data_tick, antSpec = tmp
    else:
        return None

    return data_tick, antSpec

# ============================================================
# Station configuration database
# ============================================================
STATION_CONFIG = {
    "OGA16": {
        "vmin": 0.8,
        "vmax": 1.4,

        "nAnt": 16,
        "nRow": 1,
        "nChan": 1024,
        "nOrder": 1,           # bf256:8, bf64:4, bf16:1

        "BW": 400e6,
        "flim": [400., 800.],

        "sepX": 1.0,
        "sepY": 1.0,

        "byteMeta": 64,
        "bytePack": 8256,
        "frame_per_pack": 0.5,
        "order_off": -2,

        "bitwidth": 4,
        "hdver": 2,
        "nBlock": 96,
        "blocklen": 102400,
    },
    "PYN16": {         # Not adjusted yet 
        "vmin": 0.8,
        "vmax": 1.4,

        "nAnt": 16,
        "nRow": 1,
        "nChan": 1024,
        "nOrder": 1,           # bf256:8, bf64:4, bf16:1

        "BW": 400e6,
        "flim": [300., 700.],

        "sepX": 1.0,
        "sepY": 1.0,

        "byteMeta": 64,
        "bytePack": 8256,
        "frame_per_pack": 0.5,
        "order_off": -2,

        "bitwidth": 4,
        "hdver": 2,
        "nBlock": 96,
        "blocklen": 102400,
    },
    "KMN16": {
        "vmin": 0.8,
        "vmax": 1.4,

        "nAnt": 16,
        "nRow": 1,
        "nChan": 1024,
        "nOrder": 1,           # bf256:8, bf64:4, bf16:1

        "BW": 400e6,
        "flim": [300., 700.],

        "sepX": 1.0,
        "sepY": 1.0,

        "byteMeta": 64,
        "bytePack": 8256,
        "frame_per_pack": 0.5,
        "order_off": 0,

        "bitwidth": 4,
        "hdver": 2,
        "nBlock": 160,
        "blocklen": 102400,
    }
}

def build_station_params(station_name, freq_ref):
    if station_name not in STATION_CONFIG:
        raise ValueError(f"Unknown station: {station_name}")

    cfg = STATION_CONFIG[station_name].copy()

    # ============================================================
    # Derived parameters
    # ============================================================
    cfg["nSubCh"] = cfg["nChan"] // cfg["nOrder"]

    cfg["timeFrame"] = cfg["nChan"] / cfg["BW"]

    cfg["freq"] = np.linspace(
        cfg["flim"][0], cfg["flim"][1],
        cfg["nChan"], endpoint=False
    )

    cfg["freq_ref"] = freq_ref
    cfg["lamb"] = 2.998e8 / (cfg["freq"] * 1e6)
    cfg["lamb0"] = 2.998e8 / (cfg["freq_ref"] * 1e6)

    cfg["pos1"] = np.arange(cfg["nAnt"]) * cfg["sepX"]
    cfg["pos2"] = np.arange(cfg["nRow"]) * cfg["sepY"]

    cfg["frate"] = cfg["BW"] / cfg["nChan"]
    cfg["prate"] = cfg["frate"] * 2

    return cfg

# ============================================================
# Command line options
# ============================================================
parser = argparse.ArgumentParser(
    prog="auto_beamform_OGA16_fix_ver5.0.0.py",
    usage=(
        "%(prog)s <csvfile> "
        "[--tstart YYYY-MM-DDTHH:MM:SS] "
        "[--tend YYYY-MM-DDTHH:MM:SS] "
        "[--odir DIR] [--beam N] [--nblock N] "
        "[--station {OGA16,KMN16,PYN16}] [--verbose]"
    ),
    description=(
        "Second-stage beamforming & Incoherent dedispersion tool for triggered baseband data.\n"
        "Optimized for OGA16 system using Chih-Yi's Fast Assembly Code."
    ),
    epilog=(
        "Examples:\n"
        "  python %(prog)s triggers.csv\n"
        "  python %(prog)s triggers.csv --tstart 2025-10-01T00:00:00 --tend 2025-10-01T01:00:00\n"
        "  python %(prog)s triggers.csv --beam 3 --odir results\n"
    ),
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("csvfile", help="CSV file containing event list")
parser.add_argument("--tstart", type=str, default=None,
                    help="Start time in format 'YYYY-MM-DDTHH:MM:SS' (UTC)")
parser.add_argument("--tend", type=str, default=None,
                    help="End time in format 'YYYY-MM-DDTHH:MM:SS' (UTC)")
parser.add_argument("--odir", type=str, default=".",
                    help="Output directory for .npz")
parser.add_argument("--nSum", type=int, default=400,
                    help="frame binning (default:400-->1ms)")
parser.add_argument("--nBin", type=int, default=1,
                    help="chan binning (default: 1)") 
parser.add_argument("--secWin", type=float, default=0.03,
                    help="Reading buffer before and after the pulse (default: 0.03sec)") # the dedispersed pulse will be +/-secWin wide
parser.add_argument("--freq_ref", type=float, default=300.,
                    help="bonsai trigger event time reference freq [MHz] (default: 300.0MHz)")
parser.add_argument("--beam", type=int, default=0,
                    help="Beam index to process (default: 0)") # beamID which should contain pulsar, calculated by the BURSTT beam simulator
parser.add_argument("--nblock", type=int, default=96,
                    help="Number of data blocks in a file (default: 96)")
parser.add_argument("--station", type=str, default="OGA16",
                    choices=["OGA16", "KMN16", "PYN16"],
                    help="Station name (default: OGA16)")    

# Show usage if no options
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()
scfg = build_station_params(args.station, args.freq_ref)

PACKET_SIZE = scfg["bytePack"]
meta = scfg["byteMeta"]
N_CHANNEL = scfg["nSubCh"]
FL2INT = 256 # converting float parameter to integer matrix
SCALE = 1.0 / (FL2INT * FL2INT)
no_bitmap = True

bfreq = scfg["freq"].reshape(-1,args.nBin).mean(axis=1)
timeSamp = scfg["timeFrame"] * args.nSum # sec

# ============================================================
# Post-processing & checks
# ============================================================
if args.tstart and args.tend:
    if args.tstart > args.tend:
        parser.error("--tstart must be earlier than --tend")

# Create output directory if needed
# args.odir.mkdir(parents=True, exist_ok=True)

# Clarify the csvfile position
if not args.csvfile:
    print("[ERROR] Missing required argument: csvfile\n")
    parser.print_usage(sys.stderr)
    sys.exit(1)


# ============================================================
# Load input CSV + Select specific date range
# ============================================================
try:
    dfile = args.csvfile
    df = pd.read_csv(dfile, parse_dates=["Timestamp"], comment="#")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    if df["Timestamp"].dt.tz is None:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
    else:
        df["Timestamp"] = df["Timestamp"].dt.tz_convert("UTC")
except Exception as e:
    print_log("error", f"Failed to read CSV: {e}")
    sys.exit(1)

if "EventID" not in df.columns:
    print_log("error", "Input CSV must contain 'EventID' column.")
    sys.exit(1)

if args.tstart and args.tend:
    t_start = pd.Timestamp(args.tstart, tz="UTC")
    t_end = pd.Timestamp(args.tend, tz="UTC")
    print(f"\n[INFO] Filtering events between {t_start} and {t_end} (UTC)")
    df = df[(df["Timestamp"] >= t_start) & (df["Timestamp"] <= t_end)]
elif args.tstart or args.tend:
    print("[WARN] Please specify both --tstart and --tend to enable time filtering.")
else:
    print("[INFO] No time range specified — processing all events.")

if df.empty:
    print("[WARN] No events found after filtering.")
    exit(0)

print(f"[INFO] {len(df)} events within the specified range will be processed.\n")

# Print events

for idx, row in df.iterrows():
    ev_name = row['EventID']
    idir= ev_name
    dt_event = Time(row['Timestamp'], format='datetime', scale='utc')
    # dt_event = Time(dt_event.datetime + timedelta(hours=8)) # for burstto1
    print(dt_event)
    ep_event = dt_event.to_value('unix')
    DM = row['DM']
    beamid = row['BeamID']; beamid_x = beamid%16; beamid_y = beamid//16
    #beamid_x = row['beam_x']
    #beamid_y = row['beam_y']
    print(f"{row['EventID']} | {row['Timestamp']} | DM={row['DM']:.3f} | Beam=({beamid_x},{beamid_y})")
    # cal2_dir = '/home/sdutta/fushan/cal_20250709_023000Z.check'  ####### defining delay calibration #####
    # cal2_dir = '/home/sdutta/fushan/cal_20250820_033210Z.check'  ####### defining delay calibration #####
    # cal2_dir = '/home/sdutta/fushan/cal_20250901_031906Z.check'  ####### defining delay calibration #####
    # cal2_dir = '/home/sdutta/fushan/cal_20250917_032613Z.check'  ####### defining delay calibration #####
    # cal2_dir = 'cal_20250917_032613Z.check'  ####### defining delay calibration #####
    # cal2_dir = 'cal_20251017_053050Z.check'  ####### defining delay calibration #####
    cal2_dir = None # for bf16

    #[sdutta@burstt11 calibration]$/data/kylin/241212_new_bf256/calibration/2nd_cal_2509171126/cal_20250917_032613Z.check

#    if os.path.isdir(ev_name):
#        print(f"Processing {event_name} in directory {event_dir} ...")
#        os.chdir(ev_name)


############

    files = load_bin_files(idir)
    nFile = len(files)
    print(files)

    if nFile == 0:
        print("[WARN] No files to process — skipping this directory.")
        continue
    
    
    ###### Simulated Pulse ########
    def pulse(DM, ep_event, freq, freq_ref):
        tau = 4148.8 * DM * (1/freq**2 - 1/freq_ref**2) # if freq>=freq_ref, then tau<0
        return ep_event+tau
    
    ep_pulse = pulse(DM, ep_event, scfg["freq"], scfg["freq_ref"])
    
    plt.plot(ep_pulse, scfg["freq"])
    plt.title(f"Expected Pulse of Event: {ev_name}")
    plt.savefig(f'Expected_dispersion_{ev_name}.png', dpi=300, bbox_inches='tight')
    
    ################################
    # Incoherent Dedispersion ver. # 
    ################################
    
    ## 1st beamform, identity matrix, select beamid_x
    ## 2nd beamform, pos+cal2, select beamid_y
    
    pix_win = int(args.secWin * 2 / scfg["timeFrame"])
    pix_win = (pix_win // 4) * 4
    print('pix_win:', pix_win)
    
    dd_spec = np.zeros((scfg["nChan"], pix_win), dtype=np.complex64)
    # dd_spec = np.zeros((pix_win, nAnt, nChan), dtype=np.complex64)
    print('dd_spec.shape', dd_spec.shape)
    
    nSubBin = scfg["nSubCh"]//args.nBin
    
    combined_dd_inten = []
    combined_freq = []      # To store frequencies per file
    win_len_list = []       # Keep track of dedispersion window lengths per file
    
    
    for i in range(nFile):
        fbin = files[i]
##################################        
        #sz = os.path.getsize(fbin)
        try:
            sz = os.path.getsize(fbin)   # follows symlink; raises if target missing
        except (OSError, FileNotFoundError) as e:
            print(f"Skipping broken/missing symlink: {fbin}  ({e})")
            continue
        
        # skip zero-length target files
        if sz == 0:
            print(f"Skipping (size=0): {fbin}")
            continue
############################
        packMax = (sz - scfg["byteMeta"]) // scfg["bytePack"]
        
        DATA_FILE = fbin
        with open(DATA_FILE, 'rb') as fh:
            mt = fh.read(64)
            hd = fh.read(64)
            tmp = decHeader2(hd)
            od = tmp[4] + scfg["order_off"]
            print('loading file : %s, order: %d'%(i, od))

            ep_begin = tmp[2] + 2 + (tmp[0] - tmp[4]) // scfg["nOrder"] * scfg["frame_per_pack"] * scfg["timeFrame"]
            ep_end = ep_begin + packMax // scfg["nRow"] * scfg["frame_per_pack"] * scfg["timeFrame"]
            ch_l = scfg["nSubCh"] * od
            ch_h = scfg["nSubCh"] * (od + 1) - 1
            ep_l = ep_pulse[ch_l]
            ep_h = ep_pulse[ch_h]

            p1 = int((ep_h - args.secWin - ep_begin) / (scfg["frame_per_pack"] * scfg["timeFrame"])) * scfg["nRow"]
            p2 = int((ep_l + args.secWin - ep_begin) / (scfg["frame_per_pack"] * scfg["timeFrame"])) * scfg["nRow"]

            if (p2 > packMax):
                p2 = packMax

            # fpga 全体の総パケット数を L とする
            fpga = np.memmap(fbin, dtype='u1', mode='r')
            L = (fpga.size - meta) // PACKET_SIZE   # 総パケット数

            # ----- Compute p1, p2 for ring buffer -----
            if p1 >= 0: # normal case
                nPack = (p2 - p1) // args.nSum * args.nSum
                nFrame = int(nPack/scfg["nRow"] * scfg["frame_per_pack"])
                nPack = nFrame * int(scfg["nRow"] / scfg["frame_per_pack"])

                data_off = meta + PACKET_SIZE * p1
                bitmap = fpga[data_off : data_off + nPack * PACKET_SIZE]

                if (no_bitmap):
                    bitmap = np.ones(nPack, dtype=bool)

                tick, spec = loadSpec(fh, p1, nPack, args.beam, bitmap=bitmap, bitwidth=scfg["bitwidth"], order_off=scfg["order_off"], hdver=scfg["hdver"], verbose=1, meta=meta) # spec.shape=(ntick,nChan)
            else:
                # go back to the end of ring buffer 
                start1   = L + p1
                start2   = 0
                sec1_len = L - start1
                sec2_len = p2
                total = sec1_len + sec2_len

                nPack = (total // args.nSum) * args.nSum
                nFrame = int(nPack/scfg["nRow"] * scfg["frame_per_pack"])
                nPack = nFrame * int(scfg["nRow"] / scfg["frame_per_pack"])

                # section1 & section2 
                need = nPack
                # section1
                sec1_take = min(need, sec1_len)
                # data_off1 = meta + PACKET_SIZE * start
                # part1 = fpga[data_off1 : data_off1 + sec1_take * PACKET_SIZE]
                need -= sec1_take

                if (no_bitmap):
                    bitmap  = np.ones(nPack, dtype=bool)
                    bitmap1 = np.ones(sec1_take, dtype=bool)

                tick1, spec1 = loadSpec(fh, start1, sec1_take, args.beam, bitmap=bitmap1, bitwidth=scfg["bitwidth"], order_off=scfg["order_off"], hdver=scfg["hdver"], verbose=1, meta=meta) # spec.shape=(ntick,nChan)
                # section2
                if need > 0:
                    sec2_take = min(need, sec2_len)
                    # data_off2 = meta
                    # part2 = fpga[data_off2 : data_off2 + sec2_take * PACKET_SIZE]

                    if (no_bitmap):
                        bitmap2 = np.ones(sec2_take, dtype=bool)

                    tick2, spec2 = loadSpec(fh, start2, sec2_take, args.beam, bitmap=bitmap2, bitwidth=scfg["bitwidth"], order_off=scfg["order_off"], hdver=scfg["hdver"], verbose=1, meta=meta) # spec.shape=(ntick,nChan)
                    spec = np.concatenate([spec1, spec2])
                    tick = np.concatenate([tick1, tick2])
                else:
                    spec = spec1            
                    tick = tick1

            nValid = np.count_nonzero(bitmap)
            fValid = float(nValid/nPack)

            # print(f"spec.time: {spec[0:10,0]}")
            # print(f"spec.chan: {spec[0,0:10]}")
            # # check spec
            # spectra = np.abs(spec).mean(axis=0, keepdims=False)
            # print(f"spectra.shape: {spectra.shape}") # (16,1024)
            # print(f"freq.shape: {freq.shape}") # (1024,)
            # for bm in range(nAnt):
            #     y = spectra[bm, :]
            #     y = np.maximum(y, 1e-12)   # avoid log10(0)
            #     plt.figure()
            #     plt.plot(freq, 10*np.log10(y))
            #     plt.xlabel("Frequency [Hz]")
            #     plt.ylabel("Power [dB]")
            #     plt.title(f"Beam {bm}")
            #     plt.tight_layout()
            #     plt.savefig(f"spec_{ev_name}_file{i}_beam{bm}.png")
            #     plt.close()

        
            nTime = spec.shape[0]//args.nSum
            spec = spec[:nTime * args.nSum] # (Hiroto, 2026/04/20) ver5.0.1 
            inten = np.abs(spec).reshape(nTime, args.nSum, nSubBin, args.nBin).mean(axis=(1,3)) # single beam; nAnt=1
            norm_inten = inten / np.mean(inten, axis=0, keepdims=True)                # Normalize in time

            # check spec
            spectra = inten.mean(axis=0, keepdims=False)
            print(f"spectra.shape: {spectra.shape}") # (1,1024)
            print(f"freq.shape: {scfg['freq'].shape}") # (1024,)
            print(f"bfreq.shape: {bfreq.shape}") # (1024,)
            spectra = np.maximum(spectra, 1e-12)   # avoid log10(0)
            spectra[spectra <= 1e-12] = np.nan
            plt.figure()
            # plt.plot(freq, 10*np.log10(spectra))
            plt.plot(bfreq, 10*np.log10(spectra))
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Power [dB]")
            plt.title(f"Beam {args.beam}")
            plt.tight_layout()
            plt.savefig(f"spec_{ev_name}_file{i}_beam{args.beam}.png")
            plt.close()

            #########
            # Dedispersion step (using channel-specific shifts)
            # for baseband
            #pix_start = np.array((ep_pulse[ch_l:(ch_l+nSubCh)] - ep_h) / timeFrame).astype(int)
            ## original pix_start will always be 0 at highest freq channel
            ## fix: the reference time is not ep_h (epoch time of the extracted data at the highest freq)
            ## instead, it should be computed from p1 + secWin
            ## with the added margin on p1, the pix_start here should be non-negative
            ep1 = ep_begin + (p1//scfg["nRow"])*scfg["frame_per_pack"]*scfg["timeFrame"]
            pix_start = np.array((ep_pulse[ch_l:(ch_l+scfg["nSubCh"])] - args.secWin - ep1) / scfg["timeFrame"]).astype(int)
            print('first pix:', pix_start[-1])
    
            for chi in range(scfg["nSubCh"]):
                ch = chi + ch_l
                pp = pix_start[chi]
                tmp = spec[pp:pp+pix_win,chi]
                # tmp = spec[chi,pp:pp+pix_win]
                max_len = tmp.size
                dd_spec[ch,:max_len] = tmp[:]
                # # Check tmp.shape
                # if chi == 1:
                #     print(f"tmp.shape: {tmp.shape}")
    
            ## dedispersed waterfall
            nTime = pix_win//args.nSum
            dd_len = nTime*args.nSum
            dd_inten = np.abs(dd_spec[ch_l:ch_l+scfg["nSubCh"],:dd_len]).reshape(nSubBin, args.nBin, nTime, args.nSum).mean(axis=(1,3))
            dd_inten /= dd_inten.mean(axis=1, keepdims=True) # Normalize in time
    
            #print(dd_inten)
            combined_dd_inten.append(dd_inten)
            combined_freq.append(bfreq[ch_l//args.nBin:(ch_l+scfg["nSubCh"])//args.nBin])
            
            time_per_bin_ms = args.nSum * scfg["timeFrame"] * 1000  # ms
            print(f'time_per_bin_ms={time_per_bin_ms}')
            raw_time_axis = np.arange(inten.shape[0]) * time_per_bin_ms  # ms
            dedisp_time_axis = np.arange(dd_inten.shape[1]) * time_per_bin_ms  # ms
    
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
            # 1. Dispersed Pulse (before dedispersion)
            pcm1 = axs[0].pcolormesh(raw_time_axis, bfreq[ch_l//args.nBin:(ch_l+scfg["nSubCh"])//args.nBin], 10*np.log10(norm_inten.T), shading='auto')#, vmin=vmin, vmax=vmax) #normalized plot            
            axs[0].set_title(f'Dispersed Pulse (File {i}, Order {od})')
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Frequency (MHz)')
            fig.colorbar(pcm1, ax=axs[0], label='Normalized Amplitude')
    
            # 2. Dedispersed Pulse (after dedispersion)
            pcm2 = axs[1].pcolormesh(dedisp_time_axis, bfreq[ch_l//args.nBin:(ch_l+scfg["nSubCh"])//args.nBin], 10*np.log10(dd_inten), shading='auto')        
            axs[1].set_title(f'Dedispersed Pulse (File {i}, Order {od})')
            axs[1].set_xlabel('Time (ms)')
            axs[1].set_ylabel('Frequency (MHz)')
            # fig.colorbar(pcm2, ax=axs[1], label='Normalized Amplitude')
            fig.suptitle(f"Dedispersed Pulse of Event: {ev_name}-file{i}, beam{args.beam}")
            #fig.title(f'Dedispersed Pulse of Event: {ev_name}-file{i}')
            plt.savefig(f'pulse_dedispersion_single_{ev_name}_file{i}_beam{args.beam}.png', dpi=300, bbox_inches='tight')

    # === save dedispersed baseband ===
    ep0 = ep_event - args.secWin # begin epoch of the dedispersed spec, at 400MHz
    ep  = ep_pulse - args.secWin # begin epoch of the dedispersed spec, over 400-800 MHz
    dd_time = np.arange(pix_win)*scfg["timeFrame"] # time offset of each frame
    out_name = 'baseband_beamform_dedisperse_%s.npz'%(ev_name,)
    # np.savez(out_name, freq=scfg["freq"], ep0=ep0, dd_time=dd_time, dd_spec=dd_spec, ep_event=ep_event, DM=DM, beamid_x=beamid_x, beamid_y=beamid_y) # only one beam
    np.savez(out_name, freq=scfg["freq"], ep=ep, ep0=ep0, dd_time=dd_time, dd_spec=dd_spec, ep_event=ep_event, DM=DM, beamid_x=beamid_x, beamid_y=beamid_y) # only one beam
    
    
    # === Data Processing ===
    combined_dd_inten_stack = np.vstack(combined_dd_inten)
    mean = np.nanmean(combined_dd_inten_stack, axis=1, keepdims=True)
    std  = np.nanstd (combined_dd_inten_stack, axis=1, keepdims=True)
    std[~np.isfinite(std)] = 1.0
    std[std == 0] = 1.0
    norm_combined_dd_inten_stack = (combined_dd_inten_stack - mean) / std
    clipped_inten = np.clip(norm_combined_dd_inten_stack, -4, 4)
    
    combined_freq = np.concatenate(combined_freq)

    print("std =", std)
    print("finite frac =", np.isfinite(norm_combined_dd_inten_stack).mean())
    
    # Axis edges
    freq_step = np.diff(combined_freq).mean()
    freq_edges = np.hstack([combined_freq[0] - freq_step/2, combined_freq + freq_step/2])
    time_axis = np.arange(combined_dd_inten_stack.shape[1]) * time_per_bin_ms
    time_edges = np.hstack([time_axis[0] - time_per_bin_ms/2, time_axis + time_per_bin_ms/2])
    
    # === SNR Calculation (positive) ===
    # signal = np.max(clipped_inten, axis=0)
    # noise = np.std(clipped_inten, axis=0)
    #signal = np.max((combined_dd_inten_stack-mean), axis=0)
    #noise = np.std((combined_dd_inten_stack-mean), axis=0)
    ####
    # signal = np.sum((norm_combined_dd_inten_stack), axis=0)
    # #noise = np.std((norm_combined_dd_inten_stack), axis=0)
    # noise = np.std(norm_combined_dd_inten_stack)  # single scalar noise level
    # snr_time = signal / noise
    # snr_time -= np.median(snr_time)
    #
    
    # clipped = np.clip(norm_combined_dd_inten_stack[0:256,:], 0, 10) # 400-500MHz
    clipped = np.clip(norm_combined_dd_inten_stack, 0, 10)
    signal = np.nanmean(clipped, axis=0)
    signal = np.nan_to_num(signal, nan=0.0)
    #noise = np.std((norm_combined_dd_inten_stack), axis=0)
    #noise = np.std(clipped)  # single scalar noise level
    noise = sigma_clip(signal).std()
    # if not np.isfinite(noise) or noise == 0:
    #     noise = np.nanstd(signal)
    #     if noise == 0:
    #         noise = 1.0
    if not np.isfinite(noise) or noise == 0:
        noise = 1.0
    snr_time = signal / noise
    snr_time -= np.nanmedian(snr_time)

    #####
    # # Center each time bin by subtracting mean across frequency:
    # centered = norm_combined_dd_inten_stack - norm_combined_dd_inten_stack.mean(axis=0, keepdims=True)
    # signal = np.max(centered, axis=0)
    # noise = np.std(centered, axis=0)
    # snr_time = signal / noise
    #####
    # # Centered data (already normalized)
    # centered = norm_combined_dd_inten_stack - norm_combined_dd_inten_stack.mean(axis=0, keepdims=True)
    # # Signal is now the *sum* across frequency (broadband enhancement)
    # signal = np.sum(centered, axis=0)
    # # Noise is the std of the centered data across frequency
    # noise = np.std(centered, axis=0)
    # # SNR over time
    # snr_time = signal / noise
    # # Subtract median to zero the baseline
    # snr_time -= np.median(snr_time)
    # # from scipy.ndimage import gaussian_filter1d
    # # snr_time = gaussian_filter1d(snr_time, sigma=2)
    ######

    print("clipped_inten finite frac:",
      np.isfinite(clipped_inten).mean())
    print("clipped_inten min/max:",
        np.nanmin(clipped_inten),
        np.nanmax(clipped_inten))

    print("signal min/max:",
        np.nanmin(signal),
        np.nanmax(signal))
    print("noise:", noise)

    print("snr_time finite frac:",
      np.isfinite(snr_time).mean())
    print("snr_time min/max:",
        np.nanmin(snr_time),
        np.nanmax(snr_time))

    # --- Final defense ---
    clipped_inten = np.nan_to_num(clipped_inten, nan=0.0)
    snr_time = np.nan_to_num(snr_time, nan=0.0)
    
    # === Plotting with GridSpec (3 columns: [plot, plot, colorbar]) ===
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[20, 1, 0.5], height_ratios=[4, 1], hspace=0.05, wspace=0.05)
    
    # --- Top Plot (pcolormesh) ---
    ax0 = fig.add_subplot(gs[0, 0])  # row 0, col 0
    pc = ax0.pcolormesh(time_edges, freq_edges, clipped_inten, cmap='plasma', shading='auto')
    ax0.set_title(f'Combined Dedispersed Pulse of Event: {ev_name}')
    ax0.set_ylabel('Frequency (MHz)')
    ax0.tick_params(labelbottom=False)
    
    # --- Colorbar aligned with ax0 only ---
    cax = fig.add_subplot(gs[0, 1])  # row 0, col 1
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label('Normalized Amplitude')
    
    # --- Bottom Plot (SNR) aligned to ax0 width only ---
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)  # row 1, col 0
    ax1.plot(time_axis, snr_time, color='black', linewidth=1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('SNR')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Finalize
    # plt.tight_layout()
    plt.savefig(f'pulse_dedispersion_combined_{ev_name}_ddpack.png', dpi=300)
    plt.close(fig)


import astropy.units as u
from bursttbasics.ionFR.ionFRM import ionFRM
from astropy.coordinates import FK4, FK5, ICRS, AltAz, EarthLocation, Galactic, SkyCoord
from astropy.time import Time
ionFRM(SkyCoord(ra=53.24723779*u.deg,dec=4*u.deg+34*u.arcmin+45.03*u.arcsecond),
EarthLocation(lon=42.87354227277843*u.deg,lat=-84.32021816196247*u.deg,height=1*u.earthRad),
Time('2024-01-01T16:00:00'),'../../Projects/BURSTT/IGS0OPSFIN_20261260000_01D_02H_GIM.INX')