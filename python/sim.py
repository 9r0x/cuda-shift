#!/usr/bin/env python3
import subprocess, re, argparse, os, time
import numpy as np
import pandas as pd
from PIL import Image

class Simulation:

    def __init__(self,
                 target: str,
                 N: int,
                 in_path, out_path,
                 warmup=3,
                 repeat=5) -> None:
        self.target = target
        self.N = N
        self.warmup_cycles = warmup
        self.repeat_cycles = repeat
        self.in_path = in_path
        self.out_path = out_path

    def run(self) -> float:
        time_1 = time.time()
        p = subprocess.Popen(
            [self.target, "-r", str(self.N), self.in_path, self.out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()
        time_2 = time.time()
        return (time_2 - time_1) * 1000

    def repeat(self) -> np.ndarray:
        for i in range(self.repeat_cycles):
            print(f"Repeat {i+1}/{self.repeat_cycles}...", end="\r")
            if i == 0:
                results = self.run()
            else:
                results = np.vstack((results, self.run()))

        return results

    def warmup(self) -> None:
        for i in range(self.warmup_cycles):
            print(f"Warmup {i+1}/{self.warmup_cycles}...", end="\r")
            self.run()

def generate_fake_image(s) -> None:
    """Generate a fake image of size s x s"""
    np.random.seed(0)
    image = np.random.randint(0, 255, size=(s, s, 3), dtype=np.uint8)
    image = image.astype(np.uint8)

    img = Image.fromarray(image)
    path =f"./test/{s}.jpg"
    img.save(path)
    return path

def main(target: str) -> None:
    print(f"Running {target}...")
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists("./test"):
        os.makedirs("./test")
    all_results = []
    for s in np.arange(500, 2000, 200):
        in_path = generate_fake_image(s)
        sim = Simulation(target, N=10, warmup=2, repeat=5,
                            in_path=in_path,
                            out_path="./test/out.jpg")
        sim.warmup()
        results = sim.repeat()

        average_time = np.average(results)
        ci = 1.96 * np.std(results) / np.sqrt(sim.repeat_cycles)

        print("=" * 40)
        print(f"Image dimension: {s}")
        print(f"Average Running Time: {average_time:.3f}ms")
        print(f"Confidence Interval: {ci:.3f}ms")


        all_results.append([
            s, average_time, ci
        ])

    df = pd.DataFrame(all_results,
                      columns=[
                          "Image dimension", "Average Time(ms)", "Confidence Interval"
                      ])


    df.to_csv(f"./data/images.csv", index=False)
    
    all_results = []
    for N in np.arange(10, 200, 20):
        sim = Simulation(target, N, warmup=2, repeat=5,
                          in_path="./test/500.jpg",
                          out_path="./test/out.jpg")
        sim.warmup()
        results = sim.repeat()

        average_time = np.average(results)
        ci = 1.96 * np.std(results) / np.sqrt(sim.repeat_cycles)

        print("=" * 40)
        print(f"Kernel Radius: {N}")
        print(f"Average Running Time: {average_time:.3f}ms")
        print(f"Confidence Interval: {ci:.3f}ms")

        all_results.append([
            N, average_time, ci
        ])

    df = pd.DataFrame(all_results,
                      columns=[
                          "Kernel Radius", "Average Time(ms)", "Confidence Interval"
                      ])
    df.to_csv(f"./data/kernels.csv", index=False)


if __name__ == "__main__":
    main("./cuda/bin/tiltshift")