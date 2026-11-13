import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# V3 WIRELESS LINK MODEL
# Calibrated OFDM modulation, empirical PER and ARQ
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "v3"

POLICY_FILE = RESULTS_DIR / "v3_adaptive_modulation_policy.csv"
FRAME_FILE = RESULTS_DIR / "v3_phy_statistical_validation_frames.csv"

MAX_ATTEMPTS = 4
DEFAULT_SEED = 20261115

class WirelessLink:

    def __init__(self, seed=DEFAULT_SEED):

        if not POLICY_FILE.exists():
            raise FileNotFoundError(
                f"Missing modulation policy: {POLICY_FILE}"
            )

        if not FRAME_FILE.exists():
            raise FileNotFoundError(
                f"Missing frame validation data: {FRAME_FILE}"
            )

        self.rng = np.random.default_rng(seed)

        self.policy = pd.read_csv(POLICY_FILE)
        self.frames = pd.read_csv(FRAME_FILE)

        self.frames = self.frames[
            self.frames["Estimator"] == "LMMSE"
        ].copy()

        self.frames["Packet Error"] = (
            self.frames["Errors"] > 0
        ).astype(int)

        self.per_table = (
            self.frames
            .groupby(["Modulation", "SNR (dB)"])
            .agg(
                Frames=("Packet Error", "count"),
                Packet_Errors=("Packet Error", "sum"),
                PER=("Packet Error", "mean"),
                BER=("Frame BER", "mean")
            )
            .reset_index()
        )

        self.available_snr = sorted(
            self.policy["SNR (dB)"].astype(int).unique()
        )

        self.modulation_bits = {
            "BPSK": 1,
            "QPSK": 2,
            "16-QAM": 4,
            "64-QAM": 6
        }

        print("V3 Wireless Link initialized")
        print(f"Policy points: {len(self.available_snr)}")
        print(f"Empirical PER records: {len(self.per_table)}")
        print(f"Maximum ARQ attempts: {MAX_ATTEMPTS}")

    def _lookup_snr(self, estimated_snr_db):

        eligible = [
            snr for snr in self.available_snr
            if snr <= estimated_snr_db
        ]

        if not eligible:
            return None

        return max(eligible)

    def select_modulation(self, estimated_snr_db):

        if not np.isfinite(estimated_snr_db):
            raise ValueError("Estimated SNR must be finite.")

        lookup_snr = self._lookup_snr(estimated_snr_db)

        if lookup_snr is None:
            return {
                "status": "OUTAGE",
                "modulation": None,
                "lookup_snr_db": None
            }

        row = self.policy[
            self.policy["SNR (dB)"] == lookup_snr
        ].iloc[0]

        modulation = row["Selected Modulation"]

        if modulation == "OUTAGE":
            return {
                "status": "OUTAGE",
                "modulation": None,
                "lookup_snr_db": lookup_snr
            }

        return {
            "status": "READY",
            "modulation": modulation,
            "lookup_snr_db": lookup_snr
        }

    def get_empirical_per(self, modulation, lookup_snr_db):

        row = self.per_table[
            (self.per_table["Modulation"] == modulation)
            & (self.per_table["SNR (dB)"] == lookup_snr_db)
        ]

        if row.empty:
            raise ValueError(
                "No empirical PER calibration for "
                f"{modulation} at {lookup_snr_db} dB."
            )

        return float(row.iloc[0]["PER"])

    def transmit_packet(self, estimated_snr_db):

        link = self.select_modulation(estimated_snr_db)

        if link["status"] == "OUTAGE":

            return {
                "status": "OUTAGE",
                "delivered": False,
                "modulation": None,
                "lookup_snr_db": link["lookup_snr_db"],
                "per": None,
                "attempts": 0
            }

        modulation = link["modulation"]
        lookup_snr = link["lookup_snr_db"]

        per = self.get_empirical_per(
            modulation,
            lookup_snr
        )

        delivered = False
        attempts = 0

        for attempt in range(1, MAX_ATTEMPTS + 1):

            attempts = attempt

            # Each attempt represents a new independently
            # sampled block-fading OFDM frame.
            packet_error = self.rng.random() < per

            if not packet_error:
                delivered = True
                break

        return {
            "status": "SUCCESS" if delivered else "DROP",
            "delivered": delivered,
            "modulation": modulation,
            "lookup_snr_db": lookup_snr,
            "per": per,
            "attempts": attempts
        }


if __name__ == "__main__":

    link = WirelessLink()

    test_snr_values = [12, 16, 18, 20, 24, 26, 30, 32]

    print("\nMODULATION SELECTION TEST")
    print("=========================")

    for snr in test_snr_values:

        result = link.select_modulation(snr)

        print(
            f"Estimated SNR: {snr:2d} dB | "
            f"Status: {result['status']:6s} | "
            f"Modulation: {result['modulation']} | "
            f"Calibration SNR: {result['lookup_snr_db']}"
        )

    print("\nARQ PACKET TEST")
    print("===============")

    for snr in [16, 20, 26, 30]:

        result = link.transmit_packet(snr)

        print(
            f"SNR: {snr:2d} dB | "
            f"Modulation: {result['modulation']} | "
            f"PER: {result['per']:.5f} | "
            f"Attempts: {result['attempts']} | "
            f"Delivered: {result['delivered']}"
        )
