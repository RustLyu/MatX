#include "matx_test_harness.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

extern "C" {
#include "matx/matx_log.h"
}

matx_test_entry* _matx_tests = nullptr;
int _matx_test_cnt = 0;
int _matx_test_cap = 0;
int _matx_passed = 0;
int _matx_total = 0;
const char* _matx_cur_name = nullptr;
int _matx_cur_fail = 0;

void _matx_register_test(const char* name, void (*fn)())
{
    if (_matx_test_cnt >= _matx_test_cap) {
        int new_cap = _matx_test_cap == 0 ? 256 : _matx_test_cap * 2;
        matx_test_entry* new_arr = new matx_test_entry[new_cap];
        for (int i = 0; i < _matx_test_cnt; ++i)
            new_arr[i] = _matx_tests[i];
        delete[] _matx_tests;
        _matx_tests = new_arr;
        _matx_test_cap = new_cap;
    }
    _matx_tests[_matx_test_cnt].name = name;
    _matx_tests[_matx_test_cnt].fn = fn;
    _matx_test_cnt++;
}

static int cmp_test_name(const void* a, const void* b)
{
    return strcmp(((const matx_test_entry*)a)->name,
                  ((const matx_test_entry*)b)->name);
}

int matx_run_all_tests()
{
    /* sort by name so print tests always run before read tests */
    qsort(_matx_tests, _matx_test_cnt, sizeof(matx_test_entry), cmp_test_name);

    for (int i = 0; i < _matx_test_cnt; ++i) {
        _matx_cur_name = _matx_tests[i].name;
        _matx_cur_fail = 0;
        printf("[ RUN      ] %s\n", _matx_cur_name);
        _matx_tests[i].fn();
        if (_matx_cur_fail)
            printf("[  FAILED  ] %s\n", _matx_cur_name);
        else
            printf("[       OK ] %s\n", _matx_cur_name);
    }
    printf("\n%d tests, %d assertions, %d failures\n",
           _matx_test_cnt, _matx_total, _matx_total - _matx_passed);
    delete[] _matx_tests;
    _matx_tests = nullptr;
    return (_matx_total - _matx_passed) ? 1 : 0;
}

int main() {
    std::cout << "=== initial set up ===" << std::endl;
    matx_log_init("./logs");
    matx_log_set_level(MATX_LOG_TRACE);
    MATX_TRACE("hello world! %s", "rust");
    MATX_DEBUG("hello world! %s %d", "rust", 23);
    MATX_INFO("hello world! %s %d", "rust", 23);
    MATX_WARN("hello world! %s %d", "rust", 23);
    MATX_ERROR("hello world! %s %d", "rust", 23);
    MATX_FATAL("hello world! %s %d", "rust", 23);
    MATX_FATAL("hello world! %s %.2f", "rust", 7.688);

    int result = matx_run_all_tests();

    std::cout << "=== clean ===" << std::endl;
    return result;
}