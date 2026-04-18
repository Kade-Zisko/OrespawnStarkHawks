module dataProcess (
    input logic CLK,
    input logic rst_n,
    input logic rxActive,
    input logic txReady,
    input logic [7:0] rxDATA, // input data from the esp32
    output logic [7:0] txDATA // output data to the esp32
);

// state machine enumeration definition

logic processFinish; // signal to indicate if data processing has concluded


typedef enum logic [2:0] {
    IDLE,
    PROCESS,
    WAIT,
    SEND
} FSMstate;

FSMstate state, next_state;

//

always_ff @(posedge CLK, negedge rst_n) begin : state_register
    if(!rst_n) begin
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
        SEND : next_State = IDLE;
    endcase
end



endmodule