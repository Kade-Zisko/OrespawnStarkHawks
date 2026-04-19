module UART #(
    parameter BAUD_RATE  = 9600,
    parameter FPGA_CLOCK = 100000000 // 10 MHz
) (
    input  logic CLK,
    input  logic nRST,
    input  logic rxRaw,
    input  logic [7:0] txInput,
    input  logic dataReady,
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
logic [3:0] rxcounter;
logic [7:0] txcounter;

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
        txSend <= 1;
        rxcounter <= 0;
        txcounter <= 0;
        dataRec <= 0;
    end else begin
        tstate <= next_tstate;
        rstate <= next_rstate;
        rxActive <= (next_rstate == RECEIVING);
    end
end

always_comb begin : rxStateHandler
    casez (rstate)
        RIDLE: next_rstate = dataRec ? RECEIVING : RIDLE;
        RECEIVING: next_rstate = recDone ? RDONE : RECEIVING;
        RDONE: next_rstate = RIDLE;
        default: next_rstate = RIDLE;
    endcase
end

always_comb begin : txStateHandler
    casez (tstate)
        TIDLE: next_tstate = dataSend ? SENDING : TIDLE;
        SENDING: next_tstate = sendDone ? TDONE   : SENDING;
        TDONE: next_tstate = TIDLE;
        default: next_tstate = TIDLE;
    endcase
end

logic rxLast;
always_ff @(posedge CLK, negedge nRST) begin
    if (!nRST) rxLast <= 1;
    else rxLast <= rxRaw;
end

logic startBit;
always_ff @(posedge CLK, negedge nRST) begin
    if (!nRST) startBit <= 0;
    else startBit <= (rstate == RIDLE) && (rxLast == 1) && (rxRaw == 0);
end

uartCounter #(
    .BAUD_RATE(BAUD_RATE),
    .FPGA_CLOCK(FPGA_CLOCK)
) rxCounter (
    .CLK(CLK),
    .nRST(nRST),
    .doAction(sampleBit),
    .sRST(startBit)
);

always_ff @(posedge CLK, negedge nRST) begin : rxLogic
    if (!nRST) begin
        dataRec <= 0;
        rxcounter <= 0;
        rxOutput <= 0;
    end else if (rstate == RIDLE && sampleBit) begin
        dataRec <= (rxRaw == 0);
    end else if (rstate == RECEIVING && sampleBit) begin
        if (rxcounter < 4'h8) begin
            rxOutput[rxcounter] <= rxRaw;
            rxcounter <= rxcounter + 1;
        end
    end else if (rstate == RDONE) begin
        dataRec <= 0;
        rxcounter <= 0;
    end
end

uartCounter #(
    .BAUD_RATE(BAUD_RATE),
    .FPGA_CLOCK(FPGA_CLOCK)
) txCounter (
    .CLK(CLK),
    .nRST(nRST),
    .doAction(sendBit),
    .sRST(1'b0)
);

always_ff @(posedge CLK, negedge nRST) begin : txLogic
    if (!nRST) begin
        txReady <= 1;
        txSend <= 1;
        txcounter <= 0;
    end else begin
        if(tstate == TIDLE) txReady <= !dataReady;
        else if(tstate == TDONE) txReady <= 1;
        else txReady <= 0;

        if (tstate == SENDING && sendBit) begin
            if (txcounter == 8'h0) begin
                txSend<= 0;                     
                txcounter <= txcounter + 1;
            end else if (txcounter < 8'h9) begin
                txSend<= txInput[txcounter - 1]; 
                txcounter <= txcounter + 1;
            end else if (txcounter == 8'h9) begin
                txSend <= 1;                      
                txcounter <= txcounter + 1;
            end
        end else if (tstate == TDONE) begin
            txcounter <= 0;
            txReady <= 1;
        end
    end
end

assign dataSend = (tstate == TIDLE)    && dataReady;
assign sendDone = (tstate == SENDING)  && (txcounter == 8'hA) && sendBit;
assign recDone  = (rstate == RECEIVING) && (rxcounter == 4'h8) && sampleBit;

endmodule
