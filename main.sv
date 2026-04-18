module main (
    input logic CLK,
    input logic nRST,
    input logic rxRaw, // input data from the esp32
    output logic txOutput // output data to the esp32
);

// UART data stuff
// sends -> (LOW bit for incoming data noti) -> (next 8 bits are tranfered data) -> (one last HIGH bit to indicate data stream is complete)

// Instantiate the data processing module

logic [7:0] processedData; 
logic [7:0] rxOutput;
logic txReady;
logic rxActive;

dataProcess sensorLogic (
    .CLK(CLK),
    .nRST(nRST),
    .rxData(rxOutput),
    .txData(processedData),
    .rxActive(rxActive), 
    .txReady(txReady) 
);

// Instantiate the UART module

UART uartInterface (
    .CLK(CLK),
    .nRST(nRST),
    .rxRaw(rxRaw), 
    .txInput(processedData),
    .dataReady(txReady), 
    .txReady(txReady), 
    .txSend(txOutput), 
    .rxOutput(rxOutput),
    .rxActive(rxActive)
);

endmodule