#include <gtest/gtest.h>
#include <iostream>

extern "C" {
#include "matx/matx_log.h"
}


class GlobalTestEnvironment : public testing::Environment {
public:
    void SetUp() override {
        std::cout << "=== initial set up ===" << std::endl;
        matx_log_init("./logs");
    }

    void TearDown() override {
        std::cout << "=== clean ===" << std::endl;
    }

    std::string GetGlobalResource() const {
        return global_resource;
    }

private:
    std::string global_resource;
};

static GlobalTestEnvironment* g_global_env = nullptr;

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    g_global_env = new GlobalTestEnvironment();
    testing::AddGlobalTestEnvironment(g_global_env);
    return RUN_ALL_TESTS();
}