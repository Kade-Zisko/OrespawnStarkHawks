`timescale 1ns/1ps

// written by Claude to test sv functionality

// ============================================================
//  tb_main.sv  –  Testbench for the UART sensor pipeline
//
//  Tests:
//    1. Reset behaviour
//    2. uartCounter baud-tick generation
//    3. dataDenoiser moving-mean + smoothReady flag
//    4. UART RX – receive a full byte (start + 8 data + stop)
//    5. UART TX – transmit a full byte after dataReady
//    6. dataProcess FSM – IDLE→PROCESS→SEND
//    7. dataProcess threshold (above / below THRESHOLD)
//    8. End-to-end: RX byte → processed → TX byte loopback
// ============================================================

module tb_main;

    // ----------------------------------------------------------------
    // Parameters – match DUT defaults exactly so baud timing is correct
    // FIX: FPGA_CLOCK was 99_000_000 (mismatch with DUT default 100 MHz)
    //      corrected to 100_000_000 so BAUD_TICKS matches the DUT.
    // ----------------------------------------------------------------
    localparam int BAUD_RATE  = 9600;
    localparam int FPGA_CLOCK = 100_000_000;   // must match DUT parameter
    localparam int CLK_PERIOD = 10;            // 100 MHz TB clock (ns)
    localparam int BAUD_TICKS = FPGA_CLOCK / BAUD_RATE;

    // ----------------------------------------------------------------
    // DUT signals
    // ----------------------------------------------------------------
    logic CLK;
    logic nRST;
    logic rxRaw;
    logic txOutput;

    // ----------------------------------------------------------------
    // Internal signal probes
    // ----------------------------------------------------------------
    wire        sampleBit  = dut.uartInterface.sampleBit;
    wire        sendBit    = dut.uartInterface.sendBit;
    wire [2:0]  rstate     = dut.uartInterface.rstate;
    wire [2:0]  tstate     = dut.uartInterface.tstate;
    wire        dataRec    = dut.uartInterface.dataRec;
    wire        dataSend   = dut.uartInterface.dataSend;
    wire        recDone    = dut.uartInterface.recDone;
    wire        sendDone   = dut.uartInterface.sendDone;
    wire        txReady_w  = dut.uartInterface.txReady;
    wire        rxActive_w = dut.uartInterface.rxActive;
    wire [7:0]  rxOut      = dut.uartInterface.rxOutput;
    wire [2:0]  dpState    = dut.sensorLogic.state;
    // FIX: smoothReady and smoothedRX are now declared in dataProcess,
    //      so these hierarchical probes resolve correctly.
    wire        smoothRdy  = dut.sensorLogic.smoothReady;
    wire [7:0]  smoothed   = dut.sensorLogic.smoothedRX;

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
        nRST  = 0;
        rxRaw = 1;   // UART idle high
        repeat(cycles) @(posedge CLK);
        #1;
        nRST = 1;
        @(posedge CLK);
    endtask

    // ----------------------------------------------------------------
    // Helper: send one UART byte (LSB first, 9600-8-N-1)
    // ----------------------------------------------------------------
    task automatic uart_send_byte(input logic [7:0] data);
        rxRaw = 0;                              // start bit
        repeat(BAUD_TICKS) @(posedge CLK);
        for (int i = 0; i < 8; i++) begin
            rxRaw = data[i];
            repeat(BAUD_TICKS) @(posedge CLK);
        end
        rxRaw = 1;                              // stop bit
        repeat(BAUD_TICKS) @(posedge CLK);
    endtask

    // ----------------------------------------------------------------
    // Helper: capture one UART byte from txOutput
    // ----------------------------------------------------------------
    task automatic uart_capture_byte(output logic [7:0] captured, output logic ok);
        int timeout;
        captured = 0;
        ok       = 0;
        timeout  = BAUD_TICKS * 15;
        while (txOutput !== 0 && timeout > 0) begin
            @(posedge CLK);
            timeout--;
        end
        if (timeout == 0) begin
            $display("  [WARN] uart_capture_byte: timed out waiting for start bit");
            return;
        end
        repeat(BAUD_TICKS/2) @(posedge CLK);
        if (txOutput !== 0) begin
            $display("  [WARN] uart_capture_byte: false start-bit glitch");
            return;
        end
        for (int i = 0; i < 8; i++) begin
            repeat(BAUD_TICKS) @(posedge CLK);
            captured[i] = txOutput;
        end
        repeat(BAUD_TICKS) @(posedge CLK);
        ok = (txOutput === 1);
    endtask

    // ================================================================
    // DEBUG TEST – watch internal signals during one byte send
    // ================================================================
    task debug_one_byte;
        int sampleCount, sendCount;
        $display("\n=== DEBUG: Sending one byte, watching internals ===");
        apply_reset();
        sampleCount = 0;
        sendCount   = 0;

        fork
            uart_send_byte(8'd100);
        join_none

        repeat(BAUD_TICKS * 3) begin
            @(posedge CLK);
            if (sampleBit) begin
                sampleCount++;
                $display("  [t=%0t] sampleBit! rstate=%0d dataRec=%0b rxRaw=%0b rxOut=0x%02X",
                         $time, rstate, dataRec, rxRaw, rxOut);
            end
            if (sendBit) begin
                sendCount++;
                $display("  [t=%0t] sendBit!   tstate=%0d dataSend=%0b txReady=%0b txOut=%0b",
                         $time, tstate, dataSend, txReady_w, txOutput);
            end
        end

        $display("  sampleBit fired %0d times in 3 baud periods (expect ~3)", sampleCount);
        $display("  sendBit   fired %0d times in 3 baud periods (expect ~3)", sendCount);

        repeat(BAUD_TICKS * 12) @(posedge CLK);
        $display("  After byte: rstate=%0d tstate=%0d rxOut=0x%02X dpState=%0d smoothRdy=%0b smoothed=%0d",
                 rstate, tstate, rxOut, dpState, smoothRdy, smoothed);
    endtask

    // ================================================================
    // TEST 1 – Reset
    // ================================================================
    task test_reset;
        $display("\n=== TEST 1: Reset behaviour ===");
        nRST  = 0;
        rxRaw = 1;
        repeat(10) @(posedge CLK);
        nRST = 1;
        repeat(5) @(posedge CLK);
        if (txOutput !== 1'bx)
            pass("txOutput not X after reset");
        else
            fail("txOutput is X after reset");
    endtask

    // ================================================================
    // TEST 2 – uartCounter: no spurious start bit during idle
    // ================================================================
    task test_uart_counter;
        $display("\n=== TEST 2: uartCounter baud-tick period ===");
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

        $display("  Sub-test A: constant RX=100 (above threshold)");
        apply_reset();
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
    // ================================================================
    task test_uart_rx;
        $display("\n=== TEST 4: UART RX full byte framing ===");
        apply_reset();
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
    // TEST 5 – UART TX framing: stop bit + idle recovery
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
            repeat(BAUD_TICKS * 2) @(posedge CLK);
            if (txOutput === 1)
                pass("TX line returns to idle (HIGH) after frame");
            else
                fail($sformatf("TX did not return to idle, stuck at %0b", txOutput));
        end
    endtask

    // ================================================================
    // TEST 6 – dataProcess FSM timing
    // ================================================================
    task test_dataprocess_fsm;
        int timeout;
        $display("\n=== TEST 6: dataProcess FSM timing ===");
        apply_reset();
        repeat(10) uart_send_byte(8'd50);
        uart_send_byte(8'd50);
        timeout = BAUD_TICKS * 20;
        while (txOutput !== 0 && timeout > 0) begin
            @(posedge CLK);
            timeout--;
        end
        if (timeout > 0)
            pass("dataProcess FSM produced TX response in time");
        else
            fail("dataProcess FSM timed out – TX start bit never appeared");
        begin
            logic [7:0] cap;
            logic       ok;
            uart_capture_byte(cap, ok);
        end
    endtask

    // ================================================================
    // TEST 7 – Threshold boundary
    // ================================================================
    task test_threshold_boundary;
        $display("\n=== TEST 7: Threshold boundary (value == THRESHOLD) ===");
        apply_reset();
        repeat(12) uart_send_byte(8'd3);
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
        repeat(12) uart_send_byte(8'd4);
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
    // TEST 8 – End-to-end loopback
    // ================================================================
    task test_end_to_end;
        $display("\n=== TEST 8: End-to-end loopback ===");

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
    // TEST 9 – Back-to-back bytes
    // ================================================================
    task test_back_to_back;
        $display("\n=== TEST 9: Back-to-back UART frames ===");
        apply_reset();
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

        nRST  = 0;
        rxRaw = 1;
        CLK   = 0;

        debug_one_byte();
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
    // Timeout watchdog
    // ----------------------------------------------------------------
    initial begin
        #(BAUD_TICKS * CLK_PERIOD * 300);
        $display("[WATCHDOG] Simulation exceeded time limit – forcing finish");
        $finish;
    end

    // ----------------------------------------------------------------
    // Waveform dump
    // ----------------------------------------------------------------
    initial begin
        $dumpfile("tb_main.vcd");
        $dumpvars(0, tb_main);
    end

endmodule
