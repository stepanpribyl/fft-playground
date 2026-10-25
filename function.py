import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import re
import os
from typing import List


class TxtParserError(Exception):
    pass

@dataclass
class MeasurementLine:
    i: int
    x: float
    y: float
    z: float
    s: float

@dataclass
class ScalarLine:
    i: int
    s: float

@dataclass
class ParsedData:
    date: str
    time: str
    file_name: str
    data_format: str
    measurements: List[MeasurementLine]


def parse_number(value: str) -> float:
    """
    Converts numbers with comma decimal separator into float.
    Example: '219834,000000' -> 219834.0
    """
    try:
        return float(value.replace(",", "."))
    except ValueError as e:
        raise TxtParserError(f"Invalid number format: {value}") from e


def parse_file(filepath: str, strict: bool = False) -> ParsedData:
    """
    Parses the given TXT file.

    :param filepath: path to file
    :param strict: if True, raises on bad lines; if False, skips them
    :return: ParsedData object
    """
    if not filepath.endswith(".txt"):
        raise TxtParserError(f"Expected TXT file, got '{filepath}'")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 5:
        raise TxtParserError("File too short or missing required sections")

    # --- Parse header ---
    def extract_value(prefix: str, line: str) -> str:
        if not line.startswith(prefix):
            raise TxtParserError(f"Expected '{prefix}' but got '{line}'")
        return line.split(":", 1)[1].strip()

    date = extract_value("Date", lines[0])
    time = extract_value("Time", lines[1])
    data_format = extract_value("Data Format", lines[2])

    # --- Validate column header ---
    header_line = lines[3]
    expected_cols = ["X-Axis", "Y-Axis", "Z-Axis"]

    columns = re.split(r"\s+", header_line)
    if columns != expected_cols:
        raise TxtParserError(f"Invalid column header: {columns}")

    # --- Parse measurements ---
    measurements: List[MeasurementLine] = []

    for i, line in enumerate(lines[4:], start=5):
        parts = re.split(r"\s+", line)

        if len(parts) != 3:
            if strict:
                raise TxtParserError(f"Line {i}: Expected 3 columns, got {len(parts)}")
            continue

        try:
            x = parse_number(parts[0])
            y = parse_number(parts[1])
            z = parse_number(parts[2])

            measurements.append(MeasurementLine(i, x, y, z, s=0))

        except TxtParserError:
            if strict:
                raise
            continue

    if not measurements:
        raise TxtParserError("No valid measurement data found")

    return ParsedData(
        date=date,
        time=time,
        file_name=os.path.basename(filepath),
        data_format=data_format,
        measurements=measurements,
    )


class Function:
    def __init__(self, name, time, fs):
        self.time = time
        self.fs = fs
        self.N = int(fs * time)
        self.time_set = np.array([i/fs for i in range(self.N)])
        self.data_set = np.array(self.N*[0])
        self.name = name

        self.fft_freq = abs(np.fft.rfftfreq(self.N))
        self.fft_set = abs(np.fft.rfft(self.data_set))


    def add_sine(self, A, f, phi, gamma=0):
        om = 2 * np.pi * f
        phi_rad = phi * np.pi / 180
        # print(f"OM: {om}, PH: {phi_rad}")
        sine = A * np.sin(om * self.time_set  + phi_rad) * np.e**(-gamma*self.time_set)
        self.data_set = self.data_set + sine

    def add_cosine(self, A, f, phi, gamma=0):
        om = 2 * np.pi * f
        phi_rad = phi * np.pi / 180
        # print(f"OM: {om}, PH: {phi_rad}")
        cosine = A * np.cos(om * self.time_set  + phi_rad) * np.e**(-gamma*self.time_set)
        self.data_set = self.data_set + cosine

    def add_noise(self, A):
        self.data_set += np.random.normal(0, A, self.N)  # Přidání šumu v čase

    def load_from_file(self, filepath, axis, left_crop=0):
        data = parse_file(filepath)
        measurements = data.measurements[left_crop:]
        axes = "xyz"
        assert axis in axes, "The axis is wrong!"

        self.N = len(measurements)
        self.time = self.N / self.fs
        print(f"TIME: {self.time}")
        self.time_set = np.array([i/self.fs for i in range(self.N)])
        self.data_set = np.ndarray(self.N)
        for i, line in enumerate(measurements):
            self.data_set[i] = getattr(line, axis)

    def load_from_processor(self, processor, gon_f="sine"):
        for param_set in processor.result_params:
            if gon_f=="sine":
                self.add_sine(**param_set)
            elif gon_f=="cosine":
                self.add_cosine(**param_set)

    def remove_mean(self):
        self.data_set -= self.data_set.mean()

    def fft(self):
        self.fft_freq = abs(np.fft.fftfreq(self.N, d=1/self.fs))
        self.fft_set = abs(np.fft.fft(self.data_set))

    def plot(self):
        plt.plot(self.time_set, self.data_set)
        plt.show()

    def plot_fft(self):
        plt.plot(self.fft_freq, self.fft_set)
        plt.show()