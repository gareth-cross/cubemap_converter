// Copyright 2023 Gareth Cross
#pragma once

#ifdef _MSC_VER
// Silence some warnings that libfmt can trigger w/ Microsoft compiler.
#pragma warning(push)
#pragma warning(disable : 4583)
#pragma warning(disable : 4582)
#endif  // _MSC_VER
#include <fmt/ostream.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

// Generates an exception w/ a formatted string.
template <typename... Ts>
void RaiseAssert(const char* const condition, const char* const file, const int line,
                 const char* const reason_fmt = nullptr, Ts&&... args) {
  const std::string details = reason_fmt ? fmt::format(reason_fmt, std::forward<Ts>(args)...) : "None";
  fmt::print("Assertion failed: {}\nFile: {}\nLine: {}\nDetails: {}\n", condition, file, line, details);
  std::terminate();
}

// Assertion macros.
// Based on: http://cnicholson.net/2009/02/stupid-c-tricks-adventures-in-assert
#define ASSERT_IMPL(cond, file, line, handler, ...) \
  do {                                              \
    if (!static_cast<bool>(cond)) {                 \
      handler(#cond, file, line, ##__VA_ARGS__);    \
    }                                               \
  } while (false)

// Macro to use when defining an assertion.
#define ASSERT(cond, ...) ASSERT_IMPL(cond, __FILE__, __LINE__, RaiseAssert, ##__VA_ARGS__)
