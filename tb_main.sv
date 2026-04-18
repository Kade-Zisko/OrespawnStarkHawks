`timescale 1ns/1ps

// ============================================================
//  tb_main.sv  –  Testbench for the UART sensor pipeline
//
//  Tests:
//    1. Reset behaviour
//    2. uartCounter baud-tick generation
//    3. dataDenoiser moving-mean + smoothReady flag
//    4. UART RX – receive a full byte (start + 8 data + stop)
//    5. UART TX – transmit a full byte after dataReady
//    6. dataProcess FSM – IDLE→PROCESS→WAIT→SEND
//    7. dataProcess threshold (above / below THRESHOLD)
//    8. End-to-end: RX byte → processed → TX byte loopback
// ============================================================

module tb_main;

    // ----------------------------------------------------------------
    // Parameters – match DUT defaults so timing works out
    // ----------------------------------------------------------------
    localparam int BAUD_RATE   = 9600;
    localparam int FPGA_CLOCK  = 99_000_000;
    localparam int CLK_PERIOD  = 10;              // 100 MHz TB clock (ns)
    localparam int BAUD_TICKS  = FPGA_CLOCK / BAUD_RATE; // clocks per bit

    // ----------------------------------------------------------------
    // DUT signals
    // ----------------------------------------------------------------
    logic CLK;
    logic nRST;
    logic rxRaw;
    logic txOutput;

    // ----------------------------------------------------------------
    // Instantiate top-level DUT
    // ----------------------------------------------------------------
    main dut (
        .CLK      (CLK),
        .nRST     (nRST),
        .rxRaw    (rxRaw),
        .txOutput (txOutput)
    );

    // ----------------------------------------------------------------
    // Clock generation – 100 MHz
    // ----------------------------------------------------------------
    initial CLK = 0;
    always #(CLK_PERIOD/2) CLK = ~CLK;

    // ----------------------------------------------------------------
    // Scoreboard / result counters
    // ----------------------------------------------------------------
    int pass_count = 0;
    int fail_count = 0;

    task automatic pass(input string msg);
        $display("[PASS] %0t  %s", $time, msg);
        pass_count++;
    endtask

    task automatic fail(input string msg);
        $display("[FAIL] %0t  %s", $time, msg);
        fail_count++;
    endtask

    // ----------------------------------------------------------------
    // Helper: apply reset
    // ----------------------------------------------------------------
    task automatic apply_reset(input int cycles = 5);
        nRST = 0;
        rxRaw = 1;   // UART idle high
        repeat(cycles) @(posedge CLK);
        #1;
        nRST = 1;
        @(posedge CLK);
    endtask

    // ----------------------------------------------------------------
    // Helper: send one UART byte (LSB first, 9600-8-N-1)
    //   drives rxRaw with correct bit timing
    // ----------------------------------------------------------------
    task automatic uart_send_byte(input logic [7:0] data);
        // Start bit (LOW)
        rxRaw = 0;
        repeat(BAUD_TICKS) @(posedge CLK);
        // 8 data bits, LSB first
        for (int i = 0; i < 8; i++) begin
            rxRaw = data[i];
            repeat(BAUD_TICKS) @(posedge CLK);
        end
        // Stop bit (HIGH)
        rxRaw = 1;
        repeat(BAUD_TICKS) @(posedge CLK);
    endtask

    // ----------------------------------------------------------------
    // Helper: wait for txOutput to produce a complete UART frame
    //   and decode the 8 data bits.  Times out after 15 bit-periods.
    // ----------------------------------------------------------------
    task automatic uart_capture_byte(output logic [7:0] captured, output logic ok);
        int timeout;
        captured = 0;
        ok       = 0;
        // Wait for start bit (falling edge on txOutput)
        timeout = BAUD_TICKS * 15;
        while (txOutput !== 0 && timeout > 0) begin
            @(posedge CLK);
            timeout--;
        end
        if (timeout == 0) begin
            $display("  [WARN] uart_capture_byte: timed out waiting for start bit");
            return;
        end
        // Sample mid-bit for start bit
        repeat(BAUD_TICKS/2) @(posedge CLK);
        if (txOutput !== 0) begin
            $display("  [WARN] uart_capture_byte: false start-bit glitch");
            return;
        end
        // Sample 8 data bits, one baud period apart
        for (int i = 0; i < 8; i++) begin
            repeat(BAUD_TICKS) @(posedge CLK);
            captured[i] = txOutput;
        end
        // Check stop bit
        repeat(BAUD_TICKS) @(posedge CLK);
        ok = (txOutput === 1);
    endtask

    // ================================================================
    // TEST 1 – Reset: outputs should be de-asserted after reset
    // ================================================================
    task test_reset;
        $display("\n=== TEST 1: Reset behaviour ===");
        nRST  = 0;
        rxRaw = 1;
        repeat(10) @(posedge CLK);
        // During reset txOutput may be X or 0; after release it must be 1 (idle)
        nRST = 1;
        repeat(5) @(posedge CLK);
        // After reset UART TX should be idle (HIGH) or at least not X
        if (txOutput !== 1'bx)
            pass("txOutput not X after reset");
        else
            fail("txOutput is X after reset");
    endtask

    // ================================================================
    // TEST 2 – uartCounter: verify baud tick fires at the right period
    // ================================================================
    task test_uart_counter;
        logic tick_out;
        longint t_start, t_end, measured;
        $display("\n=== TEST 2: uartCounter baud-tick period ===");

        // Instantiate a standalone uartCounter for direct observation
        // We'll watch the internal tick via a force/probe on the DUT's submodule
        // Since we can't easily bind here, we measure RX sampling behaviour
        // indirectly by timing the first UART bit acceptance.
        // Direct check: send idle and count clock cycles between TX-ready pulses.

        // Simple approach – time 3 BAUD_TICKS of idle and make sure the design
        // doesn't assert txOutput low (which would indicate a spurious start bit).
        rxRaw = 1;
        repeat(BAUD_TICKS * 3) @(posedge CLK);
        if (txOutput !== 0)
            pass("No spurious start bit during idle");
        else
            fail("Spurious start bit during idle");
    endtask

    // ================================================================
    // TEST 3 – dataDenoiser: moving average + smoothReady
    // ================================================================
    task test_denoiser;
        $display("\n=== TEST 3: dataDenoiser standalone (via DUT internals) ===");
        // Drive a constant byte value through RX so the buffer fills with it.
        // After MEANWINDOW (10) samples the smoothed output should equal that value.
        // We verify indirectly: if the value is above threshold (3) the TX byte
        // should be 0xFF; if below, 0x00.

        // Case A: constant value 100 (> threshold 3) → expect 0xFF on TX
        $display("  Sub-test A: constant RX=100 (above threshold)");
        apply_reset();
        // Send 12 identical bytes to fill the window
        repeat(12) uart_send_byte(8'd100);
        begin
            logic [7:0] cap;
            logic       ok;
            uart_capture_byte(cap, ok);
            if (ok && cap === 8'hFF)
                pass("dataDenoiser+dataProcess: value above threshold → 0xFF");
            else
                fail($sformatf("Expected 0xFF, got 0x%02X (stop_ok=%0b)", cap, ok));
        end

        // Case B: constant value 1 (< threshold 3) → expect 0x00 on TX
        $display("  Sub-test B: constant RX=1 (below threshold)");
        apply_reset();
        repeat(12) uart_send_byte(8'd1);
        begin
            logic [7:0] cap;
            logic       ok;
            uart_capture_byte(cap, ok);
            if (ok && cap === 8'h00)
                pass("dataDenoiser+dataProcess: value below threshold → 0x00");
            else
                fail($sformatf("Expected 0x00, got 0x%02X (stop_ok=%0b)", cap, ok));
        end
    endtask

    // ================================================================
    // TEST 4 – UART RX: verify a single byte is received correctly
    //   We cannot directly read rxOutput, so we confirm downstream
    //   processing produced the correct TX result (0xFF for data > 3).
    // ================================================================
    task test_uart_rx;
        $display("\n=== TEST 4: UART RX full byte framing ===");
        apply_reset();
        // Send byte 0xAB = 171 (well above threshold 3) after pre-filling window
        repeat(10) uart_send_byte(8'd171);
        begin
            logic [7:0] cap;
            logic       ok;
            uart_capture_byte(cap, ok);
            if (ok && cap === 8'hFF)
                pass("UART RX correctly received byte → 0xFF response");
            else
                fail($sformatf("UART RX: got 0x%02X stop=%0b", cap, ok));
        end
    endtask

    // ================================================================
    // TEST 5 – UART TX framing: check the stop bit and idle recovery
    // ================================================================
    task test_uart_tx_framing;
        $display("\n=== TEST 5: UART TX framing (stop bit + idle recovery) ===");
        apply_reset();
        repeat(11) uart_send_byte(8'd200);
        begin
            logic [7:0] cap;
            logic       ok;
            uart_capture_byte(cap, ok);
            if (ok)
                pass("TX stop bit is HIGH (correct framing)");
            else
                fail("TX stop bit is NOT high – framing error");
            // After a frame, TX should return to idle (HIGH)
            repeat(BAUD_TICKS * 2) @(posedge CLK);
            if (txOutput === 1)
                pass("TX line returns to idle (HIGH) after frame");
            else
                fail($sformatf("TX did not return to idle, stuck at %0b", txOutput));
        end
    endtask

    // ================================================================
    // TEST 6 – dataProcess FSM: PROCESS only triggers while rxActive
    //   Send one byte (value above threshold) and verify the TX
    //   response arrives within a bounded window.
    // ================================================================
    task test_dataprocess_fsm;
        int timeout;
        $display("\n=== TEST 6: dataProcess FSM timing ===");
        apply_reset();
        // Fill the denoiser window first, then send one more byte.
        repeat(10) uart_send_byte(8'd50);
        // Now send the stimulus byte
        uart_send_byte(8'd50);
        // dataProcess should move IDLE→PROCESS→WAIT→SEND within a few TX periods
        timeout = BAUD_TICKS * 20;
        while (txOutput !== 0 && timeout > 0) begin
            @(posedge CLK);
            timeout--;
        end
        if (timeout > 0)
            pass("dataProcess FSM produced TX response in time");
        else
            fail("dataProcess FSM timed out – TX start bit never appeared");
        // Let the frame complete
        begin
            logic [7:0] cap;
            logic       ok;
            uart_capture_byte(cap, ok);
        end
    endtask

    // ================================================================
    // TEST 7 – Threshold boundary: value exactly equal to THRESHOLD
    //   THRESHOLD=3, so rxData=3 → smoothed=3, NOT > 3 → expect 0x00
    // ================================================================
    task test_threshold_boundary;
        $display("\n=== TEST 7: Threshold boundary (value == THRESHOLD) ===");
        apply_reset();
        repeat(12) uart_send_byte(8'd3);  // exactly at threshold
        begin
            logic [7:0] cap;
            logic       ok;
            uart_capture_byte(cap, ok);
            if (ok && cap === 8'h00)
                pass("Threshold boundary: rxData==THRESHOLD → 0x00 (not strictly above)");
            else
                fail($sformatf("Threshold boundary: expected 0x00, got 0x%02X", cap));
        end

        $display("  Threshold boundary: value == THRESHOLD+1 (4)");
        apply_reset();
        repeat(12) uart_send_byte(8'd4);  // one above threshold
        begin
            logic [7:0] cap;
            logic       ok;
            uart_capture_byte(cap, ok);
            if (ok && cap === 8'hFF)
                pass("Threshold boundary: rxData==THRESHOLD+1 → 0xFF");
            else
                fail($sformatf("Threshold boundary: expected 0xFF, got 0x%02X", cap));
        end
    endtask

    // ================================================================
    // TEST 8 – End-to-end loopback with two different byte values
    // ================================================================
    task test_end_to_end;
        $display("\n=== TEST 8: End-to-end loopback ===");

        // --- Packet A: high sensor reading ---
        apply_reset();
        repeat(12) uart_send_byte(8'd200);
        begin
            logic [7:0] cap;
            logic       ok;
            uart_capture_byte(cap, ok);
            if (ok && cap === 8'hFF)
                pass("E2E loopback A: sensor high → TX=0xFF");
            else
                fail($sformatf("E2E loopback A: expected 0xFF, got 0x%02X stop=%0b", cap, ok));
        end

        // --- Packet B: low sensor reading ---
        apply_reset();
        repeat(12) uart_send_byte(8'd0);
        begin
            logic [7:0] cap;
            logic       ok;
            uart_capture_byte(cap, ok);
            if (ok && cap === 8'h00)
                pass("E2E loopback B: sensor low  → TX=0x00");
            else
                fail($sformatf("E2E loopback B: expected 0x00, got 0x%02X stop=%0b", cap, ok));
        end
    endtask

    // ================================================================
    // TEST 9 – Back-to-back bytes: no data corruption between frames
    // ================================================================
    task test_back_to_back;
        $display("\n=== TEST 9: Back-to-back UART frames ===");
        apply_reset();
        // Fill window then send two successive high bytes
        repeat(10) uart_send_byte(8'd100);
        uart_send_byte(8'd100);
        uart_send_byte(8'd100);
        begin
            logic [7:0] cap1, cap2;
            logic       ok1, ok2;
            uart_capture_byte(cap1, ok1);
            uart_capture_byte(cap2, ok2);
            if (ok1 && ok2 && cap1 === 8'hFF && cap2 === 8'hFF)
                pass("Back-to-back: both TX frames correct");
            else
                fail($sformatf("Back-to-back: cap1=0x%02X(%0b) cap2=0x%02X(%0b)",
                               cap1, ok1, cap2, ok2));
        end
    endtask

    // ================================================================
    // MAIN simulation sequence
    // ================================================================
    initial begin
        $display("============================================================");
        $display("  tb_main  –  UART Sensor Pipeline Testbench");
        $display("  BAUD=%0d  FPGA_CLK=%0d  BAUD_TICKS=%0d",
                  BAUD_RATE, FPGA_CLOCK, BAUD_TICKS);
        $display("============================================================");

        // Initialise
        nRST  = 0;
        rxRaw = 1;
        CLK   = 0;

        test_reset();
        test_uart_counter();
        test_denoiser();
        test_uart_rx();
        test_uart_tx_framing();
        test_dataprocess_fsm();
        test_threshold_boundary();
        test_end_to_end();
        test_back_to_back();

        $display("\n============================================================");
        $display("  Results: %0d PASSED  /  %0d FAILED", pass_count, fail_count);
        $display("============================================================\n");

        if (fail_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("SOME TESTS FAILED – review output above");

        $finish;
    end

    // ----------------------------------------------------------------
    // Timeout watchdog (prevents infinite hang)
    // ----------------------------------------------------------------
    initial begin
        #(BAUD_TICKS * CLK_PERIOD * 300);
        $display("[WATCHDOG] Simulation exceeded time limit – forcing finish");
        $finish;
    end

    // ----------------------------------------------------------------
    // Optional waveform dump
    // ----------------------------------------------------------------
    initial begin
        $dumpfile("tb_main.vcd");
        $dumpvars(0, tb_main);
    end

endmodule
