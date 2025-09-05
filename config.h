// config.h

#pragma once

// ================================================================
// ANSI color codes
// ================================================================

#define COLOR_RESET    "\x1b[0m"
#define COLOR_BOLD_RED "\x1b[1;31m"
#define COLOR_GREEN    "\x1b[32m"
#define COLOR_YELLOW   "\x1b[33m"
#define COLOR_ORANGE   "\x1b[33m"
#define COLOR_CYAN     "\033[36m"

// ================================================================
// CONFIGS
// ================================================================

#define MAX_LINE_WIDTH 80

constexpr int SEQ_LEN = 8192;
constexpr int PROMPT_BUFFER_SIZE = 32768;
constexpr int VOCAB_SIZE = 151936;

constexpr int DIM = 1024;
constexpr int HIDDEN_DIM = 3072;
constexpr int N_LAYERS = 28;
constexpr int N_HEADS = 16;
constexpr int N_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;

constexpr float INV_HEAD_DIM = 1.0f / HEAD_DIM;
constexpr float INV_DIM = 1.0f / DIM;

constexpr float ROPE_THETA = 1000000.0f;
constexpr float EPS = 1e-6f;

constexpr int Q_DIM =     N_HEADS * HEAD_DIM; // 16 * 128 = 2048
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; //  8 * 128 = 1024

