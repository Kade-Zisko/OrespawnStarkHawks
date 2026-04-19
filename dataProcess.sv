module dataProcess #(
    parameter THRESHOLD  = 156,
    parameter MEANWINDOW = 10
) (
    input logic CLK,
    input logic nRST,
    input logic rxActive,
    input logic txReady,
    input logic [7:0] rxData,
    output logic [7:0] txData,
    output logic dataReady
);


logic [7:0] smoothedRX;
logic [7:0] smoothedRX_reg;
logic smoothReady;
logic smoothReady_reg;
logic [7:0] rxData_reg;

typedef enum logic [2:0] {
    IDLE,
    PROCESS,
    SEND
} FSMstate;

FSMstate state, next_state;

logic lastTX;

always_ff @(posedge CLK, negedge nRST) begin : state_register
    if (!nRST) begin
        state  <= IDLE;
        lastTX <= 0;
    end else begin
        state <= next_state;
        if (state == SEND && !txReady) lastTX <= 1;
        else if (state != SEND) lastTX <= 0;
    end
end

logic lastRXActive;
always_ff @(posedge CLK, negedge nRST) begin
    if (!nRST) lastRXActive <= 0;
    else lastRXActive <= rxActive;
end
wire byteComplete = lastRXActive && !rxActive;

logic byteComplete_d;
always_ff @(posedge CLK, negedge nRST) begin
    if (!nRST) byteComplete_d <= 0;
    else byteComplete_d <= byteComplete;
end

always_comb begin : stateControl
    casez (state)
        IDLE: next_state = byteComplete ? PROCESS : IDLE;
        PROCESS: next_state = (smoothReady && txReady) ? SEND : PROCESS;
        SEND: next_state = (lastTX && txReady) ? IDLE : SEND;
        default: next_state = IDLE;
    endcase
end

logic [7:0] txData_reg;

always_ff @(posedge CLK, negedge nRST) begin
    if (!nRST) txData_reg <= 8'h00;
    else if (state == PROCESS && next_state == SEND)
        txData_reg <= smoothedRX_reg > THRESHOLD ? 8'hFF : 8'h00;
end

always_comb begin : outputLogic
    txData = txData_reg;
    dataReady = (state == SEND);
end

always_ff @(posedge CLK, negedge nRST) begin
    if (!nRST) begin
        rxData_reg <= 0;
        smoothReady_reg <= 0;
        smoothedRX_reg <= 0;
    end else begin
        if (byteComplete) begin
            rxData_reg <= rxData;
        end
        if (byteComplete_d) begin
            smoothReady_reg <= smoothReady;
            smoothedRX_reg  <= smoothedRX;
        end
    end
end

dataDenoiser denoiseData (
    .CLK(CLK),
    .nRST(nRST),
    .rxRaw(rxData_reg),
    .smooth(byteComplete_d),
    .window(MEANWINDOW),
    .rxSmoothed(smoothedRX),
    .smoothReady(smoothReady)
);

endmodule
