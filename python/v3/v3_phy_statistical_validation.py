import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import v3_phy_calibration_extended as phy

# ============================================================
# V3 INDEPENDENT PHY STATISTICAL VALIDATION
# ============================================================

SEEDS = [7, 42, 99, 123, 2026]
NUM_FRAMES = 1000
SNR_DB_VALUES = [14, 16, 18, 20, 22, 24, 26, 28, 30]
BOOTSTRAP_REPLICATES = 1000

MODULATIONS = phy.MODULATIONS
ESTIMATORS = phy.ESTIMATORS

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results" / "v3"
GRAPHS_DIR = BASE_DIR / "graphs" / "v3"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def run_validation():

    frame_records = []

    print("\nV3 PHY STATISTICAL VALIDATION")
    print("=============================")
    print(f"Independent seeds: {SEEDS}")
    print(f"Frames per seed and condition: {NUM_FRAMES}")
    print(f"SNR points: {SNR_DB_VALUES}")
    print(f"Bootstrap replicates: {BOOTSTRAP_REPLICATES}")

    for seed in SEEDS:

        print(f"\n========== SEED {seed} ==========")

        phy.rng = np.random.default_rng(seed)

        for modulation in MODULATIONS:

            k = phy.bits_per_symbol(modulation)

            print(f"\n--- {modulation} ---")

            for snr_db in SNR_DB_VALUES:

                snr_linear = 10 ** (snr_db / 10)
                noise_variance = 1 / snr_linear

                for frame in range(NUM_FRAMES):

                    bits = phy.rng.integers(
                        0, 2,
                        phy.NUM_DATA * k,
                        dtype=np.int8
                    )

                    data_symbols = phy.modulate(bits, modulation)

                    transmitted = np.zeros(
                        phy.NUM_SUBCARRIERS,
                        dtype=complex
                    )

                    transmitted[phy.PILOT_INDICES] = 1 + 0j
                    transmitted[phy.DATA_INDICES] = data_symbols

                    time_signal = (
                        np.fft.ifft(transmitted)
                        * np.sqrt(phy.NUM_SUBCARRIERS)
                    )

                    transmitted_cp = np.concatenate([
                        time_signal[-phy.CYCLIC_PREFIX:],
                        time_signal
                    ])

                    channel, true_channel = phy.generate_channel()

                    received = np.convolve(
                        transmitted_cp,
                        channel
                    )

                    noise = np.sqrt(noise_variance / 2) * (
                        phy.rng.normal(size=len(received))
                        + 1j * phy.rng.normal(size=len(received))
                    )

                    received += noise

                    received_no_cp = received[
                        phy.CYCLIC_PREFIX:
                        phy.CYCLIC_PREFIX + phy.NUM_SUBCARRIERS
                    ]

                    received_frequency = (
                        np.fft.fft(received_no_cp)
                        / np.sqrt(phy.NUM_SUBCARRIERS)
                    )

                    received_pilots = (
                        received_frequency[phy.PILOT_INDICES]
                    )

                    ls_estimate = phy.ls_estimation(
                        received_pilots
                    )

                    lmmse_estimate = phy.lmmse_estimation(
                        received_pilots,
                        noise_variance
                    )

                    estimates = {
                        "Perfect CSI": true_channel,
                        "LS": ls_estimate,
                        "LMMSE": lmmse_estimate
                    }

                    received_data = (
                        received_frequency[phy.DATA_INDICES]
                    )

                    for estimator, estimate in estimates.items():

                        data_estimate = estimate[phy.DATA_INDICES]

                        magnitude = np.maximum(
                            np.abs(data_estimate),
                            1e-10
                        )

                        safe_estimate = (
                            magnitude
                            * np.exp(1j * np.angle(data_estimate))
                        )

                        equalized = received_data / safe_estimate

                        detected_bits = phy.demodulate(
                            equalized,
                            modulation
                        )

                        errors = int(np.count_nonzero(
                            detected_bits != bits
                        ))

                        frame_records.append({
                            "Seed": seed,
                            "Modulation": modulation,
                            "Bits per Symbol": k,
                            "SNR (dB)": snr_db,
                            "Estimator": estimator,
                            "Frame": frame,
                            "Errors": errors,
                            "Bits": len(bits),
                            "Frame BER": errors / len(bits)
                        })

                print(
                    f"Seed {seed} | {modulation} | "
                    f"SNR {snr_db} dB completed"
                )

    return pd.DataFrame(frame_records)


def calculate_summary(frame_df):

    summary_records = []

    group_columns = [
        "Modulation",
        "Bits per Symbol",
        "SNR (dB)",
        "Estimator"
    ]

    bootstrap_rng = np.random.default_rng(20261113)

    for keys, group in frame_df.groupby(
        group_columns,
        sort=True
    ):

        modulation, bits_per_symbol, snr_db, estimator = keys

        frame_ber = group["Frame BER"].to_numpy()
        total_errors = int(group["Errors"].sum())
        total_bits = int(group["Bits"].sum())

        ber = total_errors / total_bits

        # Resample complete frames, not individual bits.
        n = len(frame_ber)

        bootstrap_means = np.empty(
            BOOTSTRAP_REPLICATES,
            dtype=float
        )

        for start in range(0, BOOTSTRAP_REPLICATES, 100):

            batch = min(
                100,
                BOOTSTRAP_REPLICATES - start
            )

            indices = bootstrap_rng.integers(
                0, n, size=(batch, n)
            )

            bootstrap_means[start:start + batch] = (
                frame_ber[indices].mean(axis=1)
            )

        ci_low, ci_high = np.percentile(
            bootstrap_means,
            [2.5, 97.5]
        )

        summary_records.append({
            "Modulation": modulation,
            "Bits per Symbol": bits_per_symbol,
            "SNR (dB)": snr_db,
            "Estimator": estimator,
            "Seeds": len(SEEDS),
            "Frames": n,
            "Total Bits": total_bits,
            "Total Errors": total_errors,
            "BER": ber,
            "95% CI Lower": ci_low,
            "95% CI Upper": ci_high
        })

    return pd.DataFrame(summary_records)


def save_outputs(frame_df, summary_df):

    frame_path = (
        RESULTS_DIR / "v3_phy_statistical_validation_frames.csv"
    )

    summary_path = (
        RESULTS_DIR / "v3_phy_statistical_validation_summary.csv"
    )

    frame_df.to_csv(frame_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    for modulation in MODULATIONS:

        subset_mod = summary_df[
            summary_df["Modulation"] == modulation
        ]

        plt.figure(figsize=(8, 5))

        for estimator in ESTIMATORS:

            subset = subset_mod[
                subset_mod["Estimator"] == estimator
            ].sort_values("SNR (dB)")

            plt.semilogy(
                subset["SNR (dB)"],
                subset["BER"],
                marker="o",
                label=estimator
            )

            lower_error = (
                subset["BER"] - subset["95% CI Lower"]
            ).clip(lower=0)

            upper_error = (
                subset["95% CI Upper"] - subset["BER"]
            ).clip(lower=0)

            plt.errorbar(
                subset["SNR (dB)"],
                subset["BER"],
                yerr=np.vstack([lower_error, upper_error]),
                fmt="none",
                capsize=3
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Bit Error Rate")
        plt.title(
            f"V3 Statistical BER Validation - {modulation}"
        )
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        graph_name = (
            "v3_phy_statistical_"
            + modulation.lower().replace("-", "")
            + ".png"
        )

        plt.savefig(
            GRAPHS_DIR / graph_name,
            dpi=300
        )

        plt.close()

    print("\nSTATISTICAL VALIDATION COMPLETED")
    print(f"Frame-level CSV: {frame_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Graphs: {GRAPHS_DIR}")


if __name__ == "__main__":

    frames_df = run_validation()

    frames_path = (
        RESULTS_DIR / "v3_phy_statistical_validation_frames.csv"
    )

    frames_df.to_csv(frames_path, index=False)

    print("\nFrame-level data saved before summary calculation.")

    summary_df = calculate_summary(frames_df)

    save_outputs(frames_df, summary_df)
