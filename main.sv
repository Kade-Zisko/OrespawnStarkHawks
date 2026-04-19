module main (
    input  logic CLK,
    input  logic nRST,
    input  logic rxRaw,       // input data from the ESP32
    output logic txOutput     // output data to the ESP32
);

logic [7:0] processedData;
logic [7:0] rxOutput;
logic txReady;
logic rxActive;
logic dataReady;

dataProcess sensorLogic (
    .CLK(CLK),
    .nRST(nRST),
    .rxData(rxOutput),
    .txData(processedData),
    .rxActive(rxActive),
    .txReady(txReady),
    .dataReady(dataReady)
);

UART uartInterface (
    .CLK(CLK),
    .nRST(nRST),
    .rxRaw(rxRaw),
    .txInput(processedData),
    .dataReady(dataReady),
    .txReady(txReady),
    .txSend(txOutput),
    .rxOutput(rxOutput),
    .rxActive(rxActive)
);

endmodule
