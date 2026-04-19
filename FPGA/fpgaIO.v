module fpgaIO (
    input CLK,
    input nRST,
    input rx,
    output tx
);

main mainFunc(
    .CLK(CLK),
    .nRST(nRST),
    .rxRaw(rx),
    .txOutput(tx)
);

endmodule