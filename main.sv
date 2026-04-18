module main (
    input logic CLK,
    input logic rst_n,
    input logic [9:0] ESP32_rx_in, // input data from the esp32
    output logic [9:0] ESP32_tx_out // output data to the esp32
);

// UART data stuff
// sends -> (LOW bit for incoming data noti) -> (next 8 bits are tranfered data) -> (one last HIGH bit to indicate data stream is complete)

// Instantiate the data processing module

dataProcess sensorLogic (
    .clk(clk),
    .rst_n(rst_n),
    .ESP32_rx_in(ESP32_rx_in),
    .ESP32_tx_out(ESP32_tx_out)
);



// 

endmodule