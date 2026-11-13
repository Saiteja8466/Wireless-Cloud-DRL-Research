
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# =========================================================
# CONFIGURATION
# =========================================================

SEEDS = [101, 202, 303, 404, 505]

SNR_VALUES = [0, 5, 10, 15, 20]

FORGETTING_FACTORS = [0.90, 0.95, 0.98]

NUM_FRAMES = 1000
NUM_SUBCARRIERS = 64
NUM_TAPS = 8
CP_LENGTH = 16

PILOT_SPACING = 4
PILOTS = np.arange(0, NUM_SUBCARRIERS, PILOT_SPACING)
DATA = np.setdiff1d(
    np.arange(NUM_SUBCARRIERS), PILOTS
)

SWITCH_1 = 300
SWITCH_2 = 600

CONFIRMATION_FRAMES = 5

PROFILES = {
    "Exponential": np.exp(-np.arange(NUM_TAPS)),
    "Uniform": np.ones(NUM_TAPS)
}

for name in PROFILES:
    PROFILES[name] /= np.sum(PROFILES[name])

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# =========================================================
# QPSK
# =========================================================

def qpsk_modulate(bits):

    pairs = bits.reshape(-1, 2)

    return (
        (1 - 2 * pairs[:, 0])
        + 1j * (1 - 2 * pairs[:, 1])
    ) / np.sqrt(2)


def qpsk_demodulate(symbols):

    bits = np.empty(
        (len(symbols), 2),
        dtype=np.int8
    )

    bits[:, 0] = (symbols.real < 0).astype(np.int8)
    bits[:, 1] = (symbols.imag < 0).astype(np.int8)

    return bits.reshape(-1)


# =========================================================
# CHANNEL COVARIANCE
# =========================================================

def calculate_covariance(profile):

    indices = np.arange(NUM_SUBCARRIERS)

    difference = (
        indices[:, None] - indices[None, :]
    )

    covariance = np.zeros(
        (NUM_SUBCARRIERS, NUM_SUBCARRIERS),
        dtype=complex
    )

    for tap, power in enumerate(profile):

        covariance += power * np.exp(
            -2j * np.pi * difference * tap
            / NUM_SUBCARRIERS
        )

    return covariance


COVARIANCES = {
    name: calculate_covariance(profile)
    for name, profile in PROFILES.items()
}


# =========================================================
# CHANNEL GENERATION
# =========================================================

def get_actual_profile(frame):

    if frame < SWITCH_1:
        return "Exponential"

    if frame < SWITCH_2:
        return "Uniform"

    return "Exponential"


def generate_channel(rng, profile_name):

    powers = PROFILES[profile_name]

    taps = (
        rng.normal(size=NUM_TAPS)
        + 1j * rng.normal(size=NUM_TAPS)
    ) / np.sqrt(2)

    taps *= np.sqrt(powers)

    response = np.fft.fft(
        taps, NUM_SUBCARRIERS
    )

    return taps, response


# =========================================================
# LMMSE
# =========================================================

def lmmse_estimate(
    pilot_values,
    noise_variance,
    covariance
):

    pilot_covariance = covariance[
        np.ix_(PILOTS, PILOTS)
    ]

    cross_covariance = covariance[:, PILOTS]

    matrix = (
        pilot_covariance
        + noise_variance * np.eye(len(PILOTS))
    )

    weights = np.linalg.solve(
        matrix, pilot_values
    )

    return cross_covariance @ weights


# =========================================================
# NEGATIVE LOG LIKELIHOOD
# =========================================================

def calculate_nll(
    pilot_values,
    noise_variance,
    covariance
):

    pilot_covariance = covariance[
        np.ix_(PILOTS, PILOTS)
    ]

    matrix = (
        pilot_covariance
        + noise_variance * np.eye(len(PILOTS))
    )

    sign, logdet = np.linalg.slogdet(matrix)

    if sign <= 0:
        raise ValueError("Invalid covariance matrix")

    solution = np.linalg.solve(
        matrix, pilot_values
    )

    quadratic = np.real(
        np.vdot(pilot_values, solution)
    )

    return float(logdet + quadratic)


# =========================================================
# LS INTERPOLATION
# =========================================================

def ls_estimate(pilot_values):

    return (
        np.interp(
            np.arange(NUM_SUBCARRIERS),
            PILOTS,
            pilot_values.real
        )
        + 1j * np.interp(
            np.arange(NUM_SUBCARRIERS),
            PILOTS,
            pilot_values.imag
        )
    )


# =========================================================
# MODEL SELECTION USING EWMA OF NLL DIFFERENCE
# =========================================================

class OnlineChannelSelector:

    def __init__(self, forgetting_factor):

        self.alpha = forgetting_factor

        self.smoothed_difference = 0.0

        self.initialized = False

        self.current_profile = "Exponential"

        self.candidate_profile = None
        self.candidate_count = 0

    def update(self, difference):

        # Positive difference favours Uniform.
        # Negative difference favours Exponential.

        if not self.initialized:

            self.smoothed_difference = difference
            self.initialized = True

        else:

            self.smoothed_difference = (
                self.alpha * self.smoothed_difference
                + (1 - self.alpha) * difference
            )

        proposed_profile = (
            "Uniform"
            if self.smoothed_difference > 0
            else "Exponential"
        )

        switched = False

        # Hysteresis: require consecutive proposals
        # before changing the selected profile.

        if proposed_profile == self.current_profile:

            self.candidate_profile = None
            self.candidate_count = 0

        else:

            if proposed_profile == self.candidate_profile:
                self.candidate_count += 1

            else:
                self.candidate_profile = proposed_profile
                self.candidate_count = 1

            if self.candidate_count >= CONFIRMATION_FRAMES:

                self.current_profile = proposed_profile

                self.candidate_profile = None
                self.candidate_count = 0

                switched = True

                # Do not retain old channel-model evidence
                # after a confirmed change.
                self.smoothed_difference = 0.0
                self.initialized = False

        return self.current_profile, switched


# =========================================================
# ONE SIMULATION FOR ONE SEED AND SNR
# =========================================================

def run_experiment(seed, snr_db):

    rng = np.random.default_rng(seed)

    noise_variance = 10 ** (-snr_db / 10)

    selectors = {
        alpha: OnlineChannelSelector(alpha)
        for alpha in FORGETTING_FACTORS
    }

    # Results for each forgetting factor
    error_counts = {
        alpha: {
            phase: 0
            for phase in ["Phase 1", "Phase 2", "Phase 3"]
        }
        for alpha in FORGETTING_FACTORS
    }

    nmse_sums = {
        alpha: {
            phase: 0.0
            for phase in ["Phase 1", "Phase 2", "Phase 3"]
        }
        for alpha in FORGETTING_FACTORS
    }

    accuracy_counts = {
        alpha: {
            phase: 0
            for phase in ["Phase 1", "Phase 2", "Phase 3"]
        }
        for alpha in FORGETTING_FACTORS
    }

    phase_counts = {
        "Phase 1": 0,
        "Phase 2": 0,
        "Phase 3": 0
    }

    perfect_errors = {
        phase: 0
        for phase in phase_counts
    }

    fixed_errors = {
        profile: {
            phase: 0
            for phase in phase_counts
        }
        for profile in PROFILES
    }

    ls_errors = {
        phase: 0
        for phase in phase_counts
    }

    perfect_nmse = {
        phase: 0.0
        for phase in phase_counts
    }

    fixed_nmse = {
        profile: {
            phase: 0.0
            for phase in phase_counts
        }
        for profile in PROFILES
    }

    ls_nmse = {
        phase: 0.0
        for phase in phase_counts
    }

    total_bits = {
        phase: 0
        for phase in phase_counts
    }

    histories = {
        alpha: []
        for alpha in FORGETTING_FACTORS
    }

    detection_events = {
        alpha: []
        for alpha in FORGETTING_FACTORS
    }

    # -----------------------------------------------------
    # FRAME LOOP
    # -----------------------------------------------------

    for frame in range(NUM_FRAMES):

        actual_profile = get_actual_profile(frame)

        if frame < SWITCH_1:
            phase = "Phase 1"
        elif frame < SWITCH_2:
            phase = "Phase 2"
        else:
            phase = "Phase 3"

        phase_counts[phase] += 1

        # Generate QPSK data
        bits = rng.integers(
            0, 2, len(DATA) * 2, dtype=np.int8
        )

        symbols = qpsk_modulate(bits)

        frequency_symbol = np.zeros(
            NUM_SUBCARRIERS,
            dtype=complex
        )

        frequency_symbol[PILOTS] = 1 + 0j
        frequency_symbol[DATA] = symbols

        # OFDM modulation
        time_symbol = (
            np.fft.ifft(frequency_symbol)
            * np.sqrt(NUM_SUBCARRIERS)
        )

        with_cp = np.concatenate([
            time_symbol[-CP_LENGTH:],
            time_symbol
        ])

        # Generate current channel
        taps, true_channel = generate_channel(
            rng, actual_profile
        )

        received = np.convolve(with_cp, taps)

        noise = np.sqrt(noise_variance / 2) * (
            rng.normal(size=len(received))
            + 1j * rng.normal(size=len(received))
        )

        received += noise

        received_no_cp = received[
            CP_LENGTH:
            CP_LENGTH + NUM_SUBCARRIERS
        ]

        received_frequency = (
            np.fft.fft(received_no_cp)
            / np.sqrt(NUM_SUBCARRIERS)
        )

        pilot_values = received_frequency[PILOTS]
        received_data = received_frequency[DATA]

        # -------------------------------------------------
        # FIXED AND LS ESTIMATORS
        # -------------------------------------------------

        ls = ls_estimate(pilot_values)

        fixed_estimates = {}

        for profile in PROFILES:

            fixed_estimates[profile] = lmmse_estimate(
                pilot_values,
                noise_variance,
                COVARIANCES[profile]
            )

        true_power = np.mean(
            np.abs(true_channel) ** 2
        )

        # -------------------------------------------------
        # BASELINE BER
        # -------------------------------------------------

        perfect_equalized = (
            received_data / true_channel[DATA]
        )

        perfect_detected = qpsk_demodulate(
            perfect_equalized
        )

        perfect_errors[phase] += np.count_nonzero(
            perfect_detected != bits
        )

        ls_detected = qpsk_demodulate(
            received_data / ls[DATA]
        )

        ls_errors[phase] += np.count_nonzero(
            ls_detected != bits
        )

        for profile in PROFILES:

            detected = qpsk_demodulate(
                received_data
                / fixed_estimates[profile][DATA]
            )

            fixed_errors[profile][phase] += (
                np.count_nonzero(detected != bits)
            )

        # -------------------------------------------------
        # BASELINE NMSE
        # -------------------------------------------------

        ls_nmse[phase] += (
            np.mean(np.abs(ls - true_channel) ** 2)
            / true_power
        )

        for profile in PROFILES:

            fixed_nmse[profile][phase] += (
                np.mean(
                    np.abs(
                        fixed_estimates[profile] - true_channel
                    ) ** 2
                ) / true_power
            )

        total_bits[phase] += len(bits)

        # -------------------------------------------------
        # ONLINE ADAPTIVE ESTIMATORS
        # -------------------------------------------------

        for alpha, selector in selectors.items():

            selected_profile = selector.current_profile

            # Estimate current frame using previous decision
            adaptive_estimate = lmmse_estimate(
                pilot_values,
                noise_variance,
                COVARIANCES[selected_profile]
            )

            detected = qpsk_demodulate(
                received_data / adaptive_estimate[DATA]
            )

            error_counts[alpha][phase] += (
                np.count_nonzero(detected != bits)
            )

            nmse_sums[alpha][phase] += (
                np.mean(
                    np.abs(
                        adaptive_estimate - true_channel
                    ) ** 2
                ) / true_power
            )

            if selected_profile == actual_profile:
                accuracy_counts[alpha][phase] += 1

            histories[alpha].append({
                "Frame": frame + 1,
                "Actual Profile": actual_profile,
                "Selected Profile": selected_profile,
                "Phase": phase
            })

            # Calculate current observation likelihoods
            nll_exp = calculate_nll(
                pilot_values,
                noise_variance,
                COVARIANCES["Exponential"]
            )

            nll_uniform = calculate_nll(
                pilot_values,
                noise_variance,
                COVARIANCES["Uniform"]
            )

            difference = nll_exp - nll_uniform

            # Update selector AFTER current-frame estimate
            previous_profile = selector.current_profile

            selected_next, switched = selector.update(
                difference
            )

            if switched:

                detection_events[alpha].append({
                    "Frame": frame + 1,
                    "Old Profile": previous_profile,
                    "New Profile": selected_next
                })

    # -----------------------------------------------------
    # BUILD RESULT TABLES
    # -----------------------------------------------------

    rows = []

    for phase in phase_counts:

        rows.append({
            "Seed": seed,
            "SNR (dB)": snr_db,
            "Forgetting Factor": np.nan,
            "Phase": phase,
            "Estimator": "Perfect CSI",
            "BER": perfect_errors[phase] / total_bits[phase],
            "NMSE": np.nan,
            "Selection Accuracy": np.nan
        })

        rows.append({
            "Seed": seed,
            "SNR (dB)": snr_db,
            "Forgetting Factor": np.nan,
            "Phase": phase,
            "Estimator": "LS",
            "BER": ls_errors[phase] / total_bits[phase],
            "NMSE": ls_nmse[phase] / phase_counts[phase],
            "Selection Accuracy": np.nan
        })

        for profile in PROFILES:

            rows.append({
                "Seed": seed,
                "SNR (dB)": snr_db,
                "Forgetting Factor": np.nan,
                "Phase": phase,
                "Estimator": f"Fixed {profile} LMMSE",
                "BER": (
                    fixed_errors[profile][phase]
                    / total_bits[phase]
                ),
                "NMSE": (
                    fixed_nmse[profile][phase]
                    / phase_counts[phase]
                ),
                "Selection Accuracy": np.nan
            })

        for alpha in FORGETTING_FACTORS:

            rows.append({
                "Seed": seed,
                "SNR (dB)": snr_db,
                "Forgetting Factor": alpha,
                "Phase": phase,
                "Estimator": "Improved Online LMMSE",
                "BER": (
                    error_counts[alpha][phase]
                    / total_bits[phase]
                ),
                "NMSE": (
                    nmse_sums[alpha][phase]
                    / phase_counts[phase]
                ),
                "Selection Accuracy": (
                    accuracy_counts[alpha][phase]
                    / phase_counts[phase]
                )
            })

    detection_rows = []

    for alpha in FORGETTING_FACTORS:

        events = detection_events[alpha]

        for switch_frame, old_profile, new_profile in [
            (SWITCH_1 + 1, "Exponential", "Uniform"),
            (SWITCH_2 + 1, "Uniform", "Exponential")
        ]:

            matching = [
                event for event in events
                if event["Old Profile"] == old_profile
                and event["New Profile"] == new_profile
                and event["Frame"] >= switch_frame
            ]

            if matching:

                detected_frame = matching[0]["Frame"]
                delay = detected_frame - switch_frame

            else:

                detected_frame = -1
                delay = -1

            detection_rows.append({
                "Seed": seed,
                "SNR (dB)": snr_db,
                "Forgetting Factor": alpha,
                "Switch": (
                    f"{old_profile} to {new_profile}"
                ),
                "Switch Frame": switch_frame,
                "Detection Frame": detected_frame,
                "Detection Delay": delay
            })

    history_rows = []

    for alpha in FORGETTING_FACTORS:

        for record in histories[alpha]:

            history_rows.append({
                "Seed": seed,
                "SNR (dB)": snr_db,
                "Forgetting Factor": alpha,
                **record
            })

    return rows, detection_rows, history_rows


# =========================================================
# MAIN
# =========================================================

def main():

    all_rows = []
    all_detection = []
    all_history = []

    total_runs = len(SEEDS) * len(SNR_VALUES)
    completed = 0

    print("\nIMPROVED ONLINE CHANNEL TRACKING")
    print("--------------------------------")
    print("Forgetting factors:", FORGETTING_FACTORS)
    print("Seeds:", len(SEEDS))
    print("SNR values:", SNR_VALUES)
    print("Frames per run:", NUM_FRAMES)

    for snr_db in SNR_VALUES:

        for seed in SEEDS:

            rows, detection, history = run_experiment(
                seed, snr_db
            )

            all_rows.extend(rows)
            all_detection.extend(detection)
            all_history.extend(history)

            completed += 1

            print(
                f"Progress: {completed}/{total_runs} | "
                f"SNR: {snr_db} dB | Seed: {seed}"
            )

    raw = pd.DataFrame(all_rows)
    detection = pd.DataFrame(all_detection)
    history = pd.DataFrame(all_history)

    raw.to_csv(
        RESULTS / "improved_tracking_raw.csv",
        index=False
    )

    detection.to_csv(
        RESULTS / "improved_tracking_detection.csv",
        index=False
    )

    history.to_csv(
        RESULTS / "improved_tracking_history.csv",
        index=False
    )

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    summary = (
        raw.groupby([
            "SNR (dB)",
            "Forgetting Factor",
            "Phase",
            "Estimator"
        ])
        .agg(
            Mean_BER=("BER", "mean"),
            Std_BER=("BER", "std"),
            Mean_NMSE=("NMSE", "mean"),
            Mean_Accuracy=("Selection Accuracy", "mean")
        )
        .reset_index()
    )

    # Preliminary t-based 95% CI across five seeds
    margin = (
        2.776 * summary["Std_BER"]
        / np.sqrt(len(SEEDS))
    )

    summary["BER_CI_Lower"] = np.maximum(
        0, summary["Mean_BER"] - margin
    )

    summary["BER_CI_Upper"] = np.minimum(
        1, summary["Mean_BER"] + margin
    )

    summary.to_csv(
        RESULTS / "improved_tracking_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # DETECTION SUMMARY
    # -----------------------------------------------------

    detection_summary = (
        detection.groupby([
            "SNR (dB)",
            "Forgetting Factor",
            "Switch"
        ])
        .agg(
            Mean_Delay=("Detection Delay", "mean"),
            Detection_Rate=(
                "Detection Delay",
                lambda x: np.mean(x >= 0)
            )
        )
        .reset_index()
    )

    detection_summary.to_csv(
        RESULTS / "improved_tracking_detection_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # BER GRAPH FOR EACH PHASE
    # -----------------------------------------------------

    for phase in [
        "Phase 1",
        "Phase 2",
        "Phase 3"
    ]:

        plt.figure(figsize=(10, 6))

        phase_data = summary[
            summary["Phase"] == phase
        ]

        for alpha in FORGETTING_FACTORS:

            subset = phase_data[
                (phase_data["Estimator"] == "Improved Online LMMSE")
                & (phase_data["Forgetting Factor"] == alpha)
            ].sort_values("SNR (dB)")

            plt.plot(
                subset["SNR (dB)"],
                subset["Mean_BER"],
                marker="o",
                label=f"Online alpha={alpha}"
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Mean BER")
        plt.title(
            f"Improved Online Channel Tracking - {phase}"
        )

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            RESULTS / f"improved_tracking_ber_{phase.lower().replace(' ', '_')}.png",
            dpi=300
        )

        plt.close()

    # -----------------------------------------------------
    # DETECTION DELAY GRAPH
    # -----------------------------------------------------

    for switch in detection_summary["Switch"].unique():

        plt.figure(figsize=(9, 5))

        subset = detection_summary[
            detection_summary["Switch"] == switch
        ]

        for alpha in FORGETTING_FACTORS:

            data = subset[
                subset["Forgetting Factor"] == alpha
            ].sort_values("SNR (dB)")

            plt.plot(
                data["SNR (dB)"],
                data["Mean_Delay"],
                marker="o",
                label=f"alpha={alpha}"
            )

        plt.xlabel("SNR (dB)")
        plt.ylabel("Mean Detection Delay (frames)")
        plt.title(f"Detection Delay: {switch}")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = (
            switch.lower().replace(" ", "_")
            .replace("to", "to")
        )

        plt.savefig(
            RESULTS / f"improved_tracking_delay_{filename}.png",
            dpi=300
        )

        plt.close()

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("\n--------------------------------")
    print("IMPROVED TRACKING COMPLETED")
    print("--------------------------------")

    print("\nDETECTION DELAY SUMMARY:")

    print(
        detection_summary.to_string(index=False)
    )

    print("\nADAPTIVE BER SUMMARY:")

    adaptive_summary = summary[
        summary["Estimator"] == "Improved Online LMMSE"
    ]

    print(
        adaptive_summary[
            [
                "SNR (dB)",
                "Forgetting Factor",
                "Phase",
                "Mean_BER",
                "Mean_NMSE",
                "Mean_Accuracy"
            ]
        ].to_string(index=False)
    )

    print("\nSaved files in results/")


if __name__ == "__main__":
    main()
