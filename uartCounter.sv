module uartCounter (
    input logic CLK,
    input logic nRST,
    input logic [31:0] baudRate,
    input logic [31:0] FPGAclock,
    output logic doAction
);

logic [31:0] counter; // counter to keep track of clock cycles for baud rate timing

always_ff @(posedge CLK, negedge nRST) begin
    if (!nRST) begin
        doAction <= 0;
        counter <= 0;
    end else begin
        if (counter >= (FPGAclock / baudRate)) begin
            doAction <= 1;
            counter <= 0; 
        end else if (counter >= (FPGAclock / baudRate) * 2) begin
            counter <= 0; 
        end else begin
            doAction <= 0;
            counter <= counter + 1; 
        end
        
    end
end

endmodule