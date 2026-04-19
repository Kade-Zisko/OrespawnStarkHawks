module dataDenoiser (
    input  logic        CLK,
    input  logic        nRST,
    input  logic [7:0]  rxRaw,
    input  logic [4:0]  window,
    input  logic        smooth,
    output logic [7:0]  rxSmoothed,
    output logic        smoothReady
);

logic [7:0] sampleBuffer [15:0];
logic [4:0] count;

always_ff @(posedge CLK, negedge nRST) begin
    if (!nRST) begin
        sampleBuffer <= '{default: '0};
        count        <= 0;
    end else if (smooth) begin
        sampleBuffer <= {sampleBuffer[14:0], rxRaw};
        if (count < 16) count <= count + 1;
    end
end

always_comb begin : movingMeanSmoothing
    logic [11:0] sum;
    sum = 0;
    if (count >= window) begin
        for (int i = 0; i < window; i++) begin
            sum = sum + sampleBuffer[i];
        end
        rxSmoothed  = sum / window;
        smoothReady = 1;
    end else begin
        rxSmoothed  = rxRaw;
        smoothReady = 0;
    end
end

endmodule
