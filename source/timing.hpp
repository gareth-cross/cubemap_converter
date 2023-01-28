// Copyright 2023 Gareth Cross
#pragma once
#include <chrono>

namespace timing {

// Very basic tic-toc mechanism to time to the different stages.
// Just keeps a running average of the last `NumTics` measurements for each stage.
struct SimpleTimer {
  struct Tics {
    static constexpr std::size_t NumTics = 20;

    // Zero initialize:
    Tics() { std::fill(tics.begin(), tics.end(), 0.0); }

    // Time the lambda and add it to the tics.
    template <typename F>
    void Record(F&& func) {
      const auto start = std::chrono::steady_clock::now();
      std::invoke(std::forward<F>(func));
      const auto end = std::chrono::steady_clock::now();
      const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      tics[counter % NumTics] = static_cast<double>(nanos) / 1.0e9;
      if (++counter == std::numeric_limits<std::size_t>::max()) {
        counter = 0;
      }
    }

    // Get average time in milliseconds.
    [[nodiscard]] double GetAverageMillis() const {
      const std::size_t count = std::min(counter, NumTics);
      if (count == 0) {
        return -1.0;
      }
      return std::accumulate(tics.begin(), tics.end(), 0.0) * 1000.0 / static_cast<double>(count);
    }

   private:
    std::size_t counter{0};
    std::array<double, NumTics> tics{};
  };

  enum class Stages : std::size_t { Load = 0, Unpack, Render, Pack, Write, MAX_VALUE };

  // Record the time taken in a stage.
  template <typename F>
  void Record(Stages stage, F&& func) {
    stages_[static_cast<std::size_t>(stage)].Record(std::forward<F>(func));
  }

  // Print the times.
  void Summarize() const {
    fmt::print("Times: load = {:.5f}, unpack = {:.5f}, render = {:.5f}, pack = {:.5f}, write = {:.5f} (milliseconds)\n",
               stages_[static_cast<std::size_t>(Stages::Load)].GetAverageMillis(),
               stages_[static_cast<std::size_t>(Stages::Unpack)].GetAverageMillis(),
               stages_[static_cast<std::size_t>(Stages::Render)].GetAverageMillis(),
               stages_[static_cast<std::size_t>(Stages::Pack)].GetAverageMillis(),
               stages_[static_cast<std::size_t>(Stages::Write)].GetAverageMillis());
  }

 private:
  std::array<Tics, static_cast<std::size_t>(Stages::MAX_VALUE)> stages_{};
};

}  // namespace timing
