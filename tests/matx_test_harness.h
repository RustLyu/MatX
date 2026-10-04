#ifndef MATX_TEST_HARNESS_H
#define MATX_TEST_HARNESS_H

#include <cmath>
#include <cstdio>
#include <cstdlib>

struct matx_test_entry
{
    const char* name;
    void (*fn)();
};

extern matx_test_entry* _matx_tests;
extern int _matx_test_cnt;
extern int _matx_test_cap;
extern int _matx_passed;
extern int _matx_total;
extern const char* _matx_cur_name;
extern int _matx_cur_fail;

void _matx_register_test(const char* name, void (*fn)());
int matx_run_all_tests();

inline bool _matx_check(bool passed, const char* expression, const char* file, int line)
{
    ++_matx_total;
    if (passed) {
        ++_matx_passed;
    } else {
        _matx_cur_fail = 1;
        std::fprintf(stderr, "%s:%d: failure: %s\n", file, line, expression);
    }
    return passed;
}

#define TEST(suite, name)                                                        \
    static void suite##_##name##_test();                                         \
    namespace {                                                                  \
    struct suite##_##name##_registrar                                            \
    {                                                                            \
        suite##_##name##_registrar()                                             \
        {                                                                        \
            _matx_register_test(#suite "." #name, &suite##_##name##_test);       \
        }                                                                        \
    };                                                                           \
    static suite##_##name##_registrar suite##_##name##_registrar_instance;       \
    }                                                                            \
    static void suite##_##name##_test()

#define MATX_ASSERT_BINARY(lhs, rhs, op, fatal)                                 \
    do {                                                                         \
        const auto& matx_lhs_value = (lhs);                                      \
        const auto& matx_rhs_value = (rhs);                                      \
        const bool matx_assertion_ok = matx_lhs_value op matx_rhs_value;          \
        if (!_matx_check(matx_assertion_ok, #lhs " " #op " " #rhs,             \
                         __FILE__, __LINE__) && (fatal))                         \
            return;                                                              \
    } while (0)

#define ASSERT_EQ(lhs, rhs) MATX_ASSERT_BINARY(lhs, rhs, ==, true)
#define EXPECT_EQ(lhs, rhs) MATX_ASSERT_BINARY(lhs, rhs, ==, false)
#define ASSERT_NE(lhs, rhs) MATX_ASSERT_BINARY(lhs, rhs, !=, true)
#define EXPECT_NE(lhs, rhs) MATX_ASSERT_BINARY(lhs, rhs, !=, false)

#define MATX_ASSERT_NEAR(lhs, rhs, tolerance, fatal)                            \
    do {                                                                         \
        const double matx_lhs_value = static_cast<double>(lhs);                  \
        const double matx_rhs_value = static_cast<double>(rhs);                  \
        const double matx_tolerance_value = static_cast<double>(tolerance);      \
        const bool matx_assertion_ok =                                           \
            std::fabs(matx_lhs_value - matx_rhs_value) <= matx_tolerance_value;  \
        if (!_matx_check(matx_assertion_ok, #lhs " ~= " #rhs,                   \
                         __FILE__, __LINE__) && (fatal))                         \
            return;                                                              \
    } while (0)

#define EXPECT_NEAR(lhs, rhs, tolerance) MATX_ASSERT_NEAR(lhs, rhs, tolerance, false)
#define ASSERT_NEAR(lhs, rhs, tolerance) MATX_ASSERT_NEAR(lhs, rhs, tolerance, true)

#define EXPECT_TRUE(expression)                                                 \
    do {                                                                         \
        const bool matx_assertion_ok = static_cast<bool>(expression);            \
        _matx_check(matx_assertion_ok, #expression, __FILE__, __LINE__);          \
    } while (0)

#define ASSERT_TRUE(expression)                                                 \
    do {                                                                         \
        const bool matx_assertion_ok = static_cast<bool>(expression);            \
        if (!_matx_check(matx_assertion_ok, #expression, __FILE__, __LINE__))     \
            return;                                                              \
    } while (0)

#endif
