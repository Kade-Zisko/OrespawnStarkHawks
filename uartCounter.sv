
module uartCounter #(
    parameter int BAUD_RATE  = 9600,
    parameter int FPGA_CLOCK = 100000000
) (
    input logic CLK,
    input logic nRST,
    output logic doAction,
    input logic sRST
);

logic [31:0] counter;

always_ff @(posedge CLK, negedge nRST) begin
    if (!nRST || sRST) begin
        doAction <= 0;
        counter <= 0;
    end else begin
        if (counter >= FPGA_CLOCK / BAUD_RATE) begin
            doAction <= 0;
            counter <= 0;
        end else if (counter == (FPGA_CLOCK / BAUD_RATE) / 2) begin
            doAction <= 1;
            counter <= counter + 1;
        end else begin
            doAction <= 0;
            counter <= counter + 1;
        end
    end
end

endmodule
