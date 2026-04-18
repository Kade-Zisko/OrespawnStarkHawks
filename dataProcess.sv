module dataProcess #(
    parameter THRESHOLD = 3, 
    parameter MEANWINDOW = 10
) (
    input logic CLK,
    input logic nRST,
    input logic rxActive,
    input logic txReady,
    input logic [7:0] rxData, 
    output logic [7:0] txData 
);

// state machine enumeration definition

logic processFinish;
logic [7:0] smoothedRX; 
logic smooth;
logic smoothReady_reg;

typedef enum logic [2:0] {
    IDLE,
    PROCESS,
    WAIT,
    SEND
} FSMstate;

FSMstate state, next_state;

//

always_ff @(posedge CLK, negedge nRST) begin : state_register
    if(!nRST) begin
        state <= IDLE;
    end else begin
        state <= next_state;
    end
end

// FSM to synchronize with data being input via uart RX and then output to uart TX

always_comb begin : stateControl
    casez(state)
        IDLE: next_state = rxActive ? PROCESS : IDLE;
        PROCESS: next_state = processFinish ? WAIT : PROCESS;
        WAIT : next_state = txReady ? SEND : WAIT;
        SEND : next_state = IDLE;
        default: next_state = IDLE;
    endcase
end

// Data processing logic

always_comb begin : checkThreshold
    if(state == PROCESS && smoothReady_reg) begin
        
        if(smoothedRX > THRESHOLD) begin
            txData = 8'hFF; 
        end else begin
            txData = 8'h00; 
        end
        processFinish = 1; 
        smooth = 0;
    end else if (state == PROCESS) begin
        smooth = 1;
        processFinish = 0;
        txData = 8'h00;
    end else begin
        processFinish = 0;
        smooth = 0;
        txData = 8'h00; 
    end
end

always_ff @(posedge CLK) begin : smoothReg
    smoothReady_reg <= smoothReady;
end 

// Data smoothing module 

dataDenoiser denoiseData (
    .CLK(CLK),
    .nRST(nRST),
    .rxRaw(rxData),
    .window(MEANWINDOW),
    .smooth(smooth),
    .rxSmoothed(smoothedRX),
    .smoothReady(smoothReady)
);

endmodule