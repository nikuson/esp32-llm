// mv_pie.S — int8 matrix x vector for ESP32-S3 using the PIE (Processor Instruction Extensions).
// One ee.vmulas.s8.accx = 16 int8 multiply-accumulates into the 40-bit ACCX register.

    .section .iram1, "ax"
    .align  4

// void mv_s8_pie(const int8_t *W, const int8_t *x, int32_t *acc, int rows, int n16)
//   a2 = W (16-aligned rows), a3 = x (16-aligned), a4 = acc, a5 = rows, a6 = cols/16 (>= 1)
    .global mv_s8_pie
    .type   mv_s8_pie, @function
mv_s8_pie:
    entry   a1, 32
    beqz    a5, 9f
    addi    a6, a6, -1              // iterations after the primed pair
1:
    mov     a7, a3
    ee.zero.accx
    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a7, 16
    loopnez a6, 2f
    ee.vmulas.s8.accx.ld.ip  q0, a2, 16, q0, q1   // MAC current pair, prefetch next W block
    ee.vld.128.ip   q1, a7, 16
2:
    ee.vmulas.s8.accx  q0, q1
    nop
    nop
    rur.accx_0  a8
    s32i    a8, a4, 0
    addi    a4, a4, 4
    addi    a5, a5, -1
    bnez    a5, 1b
9:
    retw.n
    .size   mv_s8_pie, . - mv_s8_pie

// void mv_s8_pie_64(const int8_t *W, const int8_t *x, int32_t *acc, int rows)
//   cols == 64: the whole activation vector lives in q4..q7 for the entire matrix,
//   each row is 4 loads + 4 MACs, two rows are interleaved to hide load latency.
    .global mv_s8_pie_64
    .type   mv_s8_pie_64, @function
mv_s8_pie_64:
    entry   a1, 32
    ee.vld.128.ip   q4, a3, 16
    ee.vld.128.ip   q5, a3, 16
    ee.vld.128.ip   q6, a3, 16
    ee.vld.128.ip   q7, a3, 16
    beqz    a5, 9f
1:
    ee.zero.accx
    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a2, 16
    ee.vmulas.s8.accx.ld.ip  q2, a2, 16, q0, q4
    ee.vmulas.s8.accx.ld.ip  q3, a2, 16, q1, q5
    ee.vmulas.s8.accx  q2, q6
    ee.vmulas.s8.accx  q3, q7
    nop
    nop
    rur.accx_0  a8
    s32i    a8, a4, 0
    addi    a4, a4, 4
    addi    a5, a5, -1
    bnez    a5, 1b
9:
    retw.n
    .size   mv_s8_pie_64, . - mv_s8_pie_64
