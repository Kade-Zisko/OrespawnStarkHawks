module UART #(
    parameter BAUD_RATE = 9600, // Needs to be set in the esp32 code with the serial() function
    parameter FPGA_CLOCK = 99000000 // System clock should be 100 MHZ but NEED TO VERIFY
) (
    input logic CLK,
    input logic nRST,
    input logic rxRaw,
    input logic [7:0] txInput,
    input logic dataReady,
    output logic txReady,
    output logic txSend,
    output logic [7:0] rxOutput,
    output logic rxActive
);

logic dataRec;
logic dataSend;
logic recDone;
logic sendDone;
logic sampleBit;
logic sendBit;
logic [3:0] rxcounter ; // counter to see if 8 bits have been recieved
logic [7:0] txcounter; // counter to see if 10 bits have been sent

typedef enum logic [2:0] {
    RIDLE,
    RECEIVING,
    RDONE
} rxFSM;

typedef enum logic [2:0] {
    TIDLE,
    SENDING,
    TDONE
} txFSM;

rxFSM rstate, next_rstate;

txFSM tstate, next_tstate;

always_ff @(posedge CLK, negedge nRST) begin : state_register
    if (!nRST) begin
        rstate <= RIDLE;
        tstate <= TIDLE;
        rxActive <= 0;
    end else begin
        tstate <= next_tstate;
        rstate <= next_rstate;
        rxActive <= (rstate == RECEIVING);
    end
end

always_comb begin : rxStateHandler
    casez(rstate)
        RIDLE : next_rstate = dataRec ? RECEIVING : RIDLE;
        RECEIVING : next_rstate = recDone ? RDONE : RECEIVING;
        RDONE : next_rstate = RIDLE;
        default: next_rstate = RIDLE;
    endcase
end

always_comb begin : txStateHandler
    casez(tstate)
        TIDLE : next_tstate = dataSend ? SENDING : TIDLE;
        SENDING : next_tstate = sendDone ? TDONE : SENDING;
        TDONE : next_tstate = TIDLE;
        default: next_tstate = TIDLE;
    endcase
end

uartCounter rxCounter (
    .CLK(CLK),
    .nRST(nRST),
    .baudRate(BAUD_RATE),
    .doAction(sampleBit),
    .FPGAclock(FPGA_CLOCK)
);

always_ff @(posedge CLK) begin : rxLogic
    if(rstate == RIDLE && sampleBit) begin
        dataRec <= rxRaw == 0;
    end else if (rstate == RECEIVING && sampleBit) begin
        if(rxcounter == 4'h8) begin
            recDone <= rxRaw == 1; // check for stop bit
        end else begin
            rxOutput[rxcounter] <= rxRaw;
            rxcounter <= rxcounter + 1;
        end
    end else begin
        dataRec <= 0;
        recDone <= 0;
        rxcounter <= 0;
    end
end

uartCounter txCounter (
    .CLK(CLK),
    .nRST(nRST),
    .baudRate(BAUD_RATE),
    .doAction(sendBit),
    .FPGAclock(FPGA_CLOCK)
);

always_ff @(posedge CLK) begin : txLogic
    if(tstate == TIDLE && sendBit) begin
        if(dataReady) begin
            dataSend <= 1;
            txReady <= 0;
        end else begin
            dataSend <= 0;
            txReady <= 1;
        end
    end else if (tstate == SENDING && sendBit) begin
        if (txcounter == 5'h0) begin
            txSend <= 0; // start bit
            txcounter <= txcounter + 1;
        end else if (txcounter > 0 && txcounter < 5'h9) begin
            txSend <= txInput[txcounter - 1]; // data bits
            txcounter <= txcounter + 1;
        end else if (txcounter == 5'h9) begin
            txSend <= 1; // stop bit
            txcounter <= txcounter + 1;
            sendDone <= 1;
        end
    end else begin
        dataSend <= 0;
        sendDone <= 0;
        txcounter <= 0;
        txReady <= 1;
    end
end

endmodule