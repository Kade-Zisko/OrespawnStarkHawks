`timescale 1ns/1ps

// ============================================================
//  tb_esp32_stream.sv
//
//  Simulates a noisy sensor data stream coming from an ESP32
//  over UART into the `main` DUT, and checks the TX response.
//
//  Data model:
//    - Sensor mean = 100
//    - Uniform noise in [-NOISE_AMP, +NOISE_AMP]
//    - THRESHOLD = 3, so expected TX = 0xFF for every byte
//      (every sample stays well above threshold after smoothing)
//
//  Checks:
//    1. Per-byte TX correctness (once denoiser is warm)
//    2. RX -> TX latency (measured in clock cycles)
//    3. Visual stream: prints every RX byte and every TX byte
//       with running stats
//
//  How the stream works:
//    The testbench acts as the ESP32. It drives rxRaw with
//    UART-framed bytes back-to-back. A separate always-block
//    watches txOutput and reconstructs the received byte.
// ============================================================

module tb_esp32_stream;

    // ----------------------------------------------------------------
    // Parameters — match DUT defaults
    // ----------------------------------------------------------------
    localparam int BAUD_RATE  = 9600;
    localparam int FPGA_CLOCK = 100_000_000;
    localparam int CLK_PERIOD = 10;                // 100 MHz
    localparam int BAUD_TICKS = FPGA_CLOCK / BAUD_RATE;

    localparam int STREAM_LEN   = 100;             // number of bytes
    localparam int SENSOR_MEAN  = 100;             // matches threshold
    localparam int NOISE_AMP    = 8;               // +/- noise span
    localparam int THRESHOLD    = 3;               // DUT default
    localparam int MEAN_WINDOW  = 10;              // DUT default
    localparam int WARMUP_BYTES = MEAN_WINDOW + 2; // skip before checking

    // ----------------------------------------------------------------
    // DUT signals
    // ----------------------------------------------------------------
    logic CLK;
    logic nRST;
    logic rxRaw;
    logic txOutput;

    // ----------------------------------------------------------------
    // Instantiate DUT
    // ----------------------------------------------------------------
    main dut (
        .CLK      (CLK),
        .nRST     (nRST),
        .rxRaw    (rxRaw),
        .txOutput (txOutput)
    );

    // ----------------------------------------------------------------
    // Clock
    // ----------------------------------------------------------------
    initial CLK = 0;
    always #(CLK_PERIOD/2) CLK = ~CLK;

    // ----------------------------------------------------------------
    // Stream storage — what the "ESP32" sent and what we expect back
    // ----------------------------------------------------------------
    logic [7:0] stream_data  [STREAM_LEN];
    logic [7:0] expected_tx  [STREAM_LEN];   // 0xFF or 0x00 per byte
    logic [7:0] captured_tx  [STREAM_LEN];
    int         tx_timestamp [STREAM_LEN];   // sim time TX finished
    int         rx_timestamp [STREAM_LEN];   // sim time RX finished

    int pass_count = 0;
    int fail_count = 0;
    int tx_capture_idx = 0;

    // ----------------------------------------------------------------
    // Simple PRNG — repeatable noisy sensor
    // ----------------------------------------------------------------
    int seed = 32'hCAFEBABE;

    function automatic int signed noise_sample();
        int r;
        r = ($random(seed) % (2*NOISE_AMP + 1));
        if (r < 0) r = r + (2*NOISE_AMP + 1);
        return r - NOISE_AMP;      // range: [-NOISE_AMP, +NOISE_AMP]
    endfunction

    // ----------------------------------------------------------------
    // Pre-generate the stream so we know the "ground truth"
    // ----------------------------------------------------------------
    task automatic generate_stream;
        int         v;
        int signed  noise;
        real        sum;
        begin
            sum = 0.0;
            for (int i = 0; i < STREAM_LEN; i++) begin
                noise = noise_sample();
                v = SENSOR_MEAN + noise;
                if (v < 0)   v = 0;
                if (v > 255) v = 255;
                stream_data[i] = v[7:0];

                // Expected output is based on the moving mean over the
                // last MEAN_WINDOW samples. With mean=100 and small
                // noise this is always > THRESHOLD, so 0xFF.
                sum += v;
                if (i >= MEAN_WINDOW - 1) begin
                    real mean_val;
                    mean_val = 0.0;
                    for (int k = i - MEAN_WINDOW + 1; k <= i; k++)
                        mean_val += stream_data[k];
                    mean_val = mean_val / MEAN_WINDOW;
                    expected_tx[i] = (mean_val > THRESHOLD) ? 8'hFF : 8'h00;
                end else begin
                    expected_tx[i] = 8'hxx;   // not checked during warmup
                end
            end
            $display("[GEN] Stream generated: %0d bytes, mean=%0d, noise=+/-%0d",
                      STREAM_LEN, SENSOR_MEAN, NOISE_AMP);
        end
    endtask

    // ----------------------------------------------------------------
    // ESP32 UART transmitter — send one byte, 8-N-1, LSB first
    // ----------------------------------------------------------------
    task automatic esp32_send_byte(input logic [7:0] data, input int idx);
        rxRaw = 0;                              // start bit
        repeat(BAUD_TICKS) @(posedge CLK);
        for (int i = 0; i < 8; i++) begin
            rxRaw = data[i];
            repeat(BAUD_TICKS) @(posedge CLK);
        end
        rxRaw = 1;                              // stop bit
        repeat(BAUD_TICKS) @(posedge CLK);
        rx_timestamp[idx] = $time;
        $display("[RX ->DUT t=%0t] byte[%0d] = 0x%02X (%0d)",
                 $time, idx, data, data);
    endtask

    // ----------------------------------------------------------------
    // TX-side background monitor
    // Watches txOutput continuously, reconstructs each frame,
    // stores into captured_tx[] in order of arrival.
    // ----------------------------------------------------------------
    task automatic tx_monitor;
        logic [7:0] byte_val;
        forever begin
            // Wait for start bit (falling edge to 0)
            @(negedge txOutput);
            // Skip to middle of start bit
            repeat(BAUD_TICKS/2) @(posedge CLK);
            if (txOutput === 0) begin
                // Sample 8 data bits
                for (int i = 0; i < 8; i++) begin
                    repeat(BAUD_TICKS) @(posedge CLK);
                    byte_val[i] = txOutput;
                end
                // Stop bit
                repeat(BAUD_TICKS) @(posedge CLK);
                if (tx_capture_idx < STREAM_LEN) begin
                    captured_tx[tx_capture_idx]  = byte_val;
                    tx_timestamp[tx_capture_idx] = $time;
                    $display("[TX <-DUT t=%0t] byte[%0d] = 0x%02X (stop=%0b)",
                             $time, tx_capture_idx, byte_val, txOutput);
                    tx_capture_idx++;
                end
            end
        end
    endtask

    // ----------------------------------------------------------------
    // Apply reset
    // ----------------------------------------------------------------
    task automatic apply_reset;
        nRST  = 0;
        rxRaw = 1;
        repeat(10) @(posedge CLK);
        nRST = 1;
        @(posedge CLK);
    endtask

    // ----------------------------------------------------------------
    // Result reporting
    // ----------------------------------------------------------------
    task automatic report_results;
        int          checked;
        int          correct;
        longint      lat_sum;
        int          lat_min;
        int          lat_max;
        int          lat_cycles;
        int          lat_checked;
        begin
            checked = 0;
            correct = 0;
            lat_sum = 0;
            lat_min = 32'h7FFFFFFF;
            lat_max = 0;
            lat_checked = 0;

            $display("\n============================================================");
            $display(" STREAM RESULTS");
            $display("============================================================");
            $display(" Idx  RXdata   Expect   Got     RXtime(ns)    TXtime(ns)    Latency(cycles)");
            $display(" ---  ------   ------   ------  -----------   -----------   ---------------");

            for (int i = 0; i < tx_capture_idx && i < STREAM_LEN; i++) begin
                string verdict;
                if (i < WARMUP_BYTES) begin
                    verdict = "warmup";
                end else begin
                    checked++;
                    if (captured_tx[i] === expected_tx[i]) begin
                        correct++;
                        verdict = "OK";
                    end else begin
                        verdict = "FAIL";
                    end
                end

                lat_cycles = (tx_timestamp[i] - rx_timestamp[i]) / CLK_PERIOD;
                if (i >= WARMUP_BYTES) begin
                    lat_sum += lat_cycles;
                    if (lat_cycles < lat_min) lat_min = lat_cycles;
                    if (lat_cycles > lat_max) lat_max = lat_cycles;
                    lat_checked++;
                end

                $display(" %3d   0x%02X     0x%02X     0x%02X    %10d    %10d    %8d   %s",
                         i, stream_data[i], expected_tx[i], captured_tx[i],
                         rx_timestamp[i], tx_timestamp[i], lat_cycles, verdict);
            end

            $display("\n============================================================");
            $display(" SUMMARY");
            $display("============================================================");
            $display(" Bytes sent:        %0d", STREAM_LEN);
            $display(" TX frames seen:    %0d", tx_capture_idx);
            $display(" Bytes checked:     %0d  (skipped first %0d warmup)",
                     checked, WARMUP_BYTES);
            $display(" Correct:           %0d / %0d", correct, checked);
            if (lat_checked > 0) begin
                $display(" Latency (cycles):  min=%0d  max=%0d  avg=%0d",
                         lat_min, lat_max, lat_sum / lat_checked);
                $display(" Latency (us):      min=%0.2f  max=%0.2f  avg=%0.2f",
                         lat_min  * CLK_PERIOD / 1000.0,
                         lat_max  * CLK_PERIOD / 1000.0,
                         (lat_sum * CLK_PERIOD) / (lat_checked * 1000.0));
            end

            if (checked > 0 && correct == checked)
                $display(" RESULT: ALL CORRECT");
            else
                $display(" RESULT: %0d MISMATCHES", checked - correct);
            $display("============================================================\n");

            pass_count = correct;
            fail_count = checked - correct;
        end
    endtask

    // ----------------------------------------------------------------
    // Main sequence
    // ----------------------------------------------------------------
    initial begin
        $display("============================================================");
        $display("  tb_esp32_stream — simulated ESP32 sensor data stream");
        $display("  BAUD=%0d  FPGA_CLK=%0d  BAUD_TICKS=%0d",
                  BAUD_RATE, FPGA_CLOCK, BAUD_TICKS);
        $display("  Stream: %0d bytes, mean=%0d, noise=+/-%0d",
                  STREAM_LEN, SENSOR_MEAN, NOISE_AMP);
        $display("============================================================\n");

        nRST  = 0;
        rxRaw = 1;

        generate_stream();

        fork
            tx_monitor();
        join_none

        apply_reset();

        // Small idle gap before first byte (ESP32 boot delay)
        repeat(BAUD_TICKS) @(posedge CLK);

        // Stream the bytes back-to-back
        for (int i = 0; i < STREAM_LEN; i++) begin
            esp32_send_byte(stream_data[i], i);
        end

        // Drain: wait for any in-flight TX frames to finish.
        // TX runs slightly slower than RX (each takes ~10 baud periods
        // plus FSM overhead), so after STREAM_LEN bytes the TX can be
        // several frames behind. Give it plenty of slack.
        repeat(BAUD_TICKS * 11 * 20) @(posedge CLK);

        report_results();
        $finish;
    end

    // ----------------------------------------------------------------
    // Watchdog
    // ----------------------------------------------------------------
    initial begin
        // Enough time for STREAM_LEN bytes + generous TX drain
        #(BAUD_TICKS * CLK_PERIOD * (STREAM_LEN * 11 + 300));
        $display("[WATCHDOG] Simulation exceeded time limit");
        report_results();
        $finish;
    end

    // ----------------------------------------------------------------
    // Waveform dump
    // ----------------------------------------------------------------
    initial begin
        $dumpfile("tb_esp32_stream.vcd");
        $dumpvars(0, tb_esp32_stream);
    end

endmodule
