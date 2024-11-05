############################################################
## Loop through different optimizations and execute synthesis
## C/RTL cosimulation, and export RTL to VHDL with a custom name
############################################################

# Open the project
open_project matrix_multiplication

# Define the top-level function (this will change dynamically)
set top_function ""

# Define optimizations in a list
set full_exec 1
set algorithms {0 1 2 3}
set optimizations {0 1 2 3}
set sizes {8 16 32 64 128 256}

# Iterate over each optimization
foreach algorithm ${algorithms} {
    foreach optimization ${optimizations} {
        foreach size ${sizes} {
            # Set the top function based on the optimization (matrix_mult_0, matrix_mult_1, ...)
            set top_function "matrix_mult_${algorithm}_${optimization}"
            puts "INFO: TCL ${top_function} with SIZE=${size}."
            puts "================================================================================\n"

            # Set top-level function in HLS
            set_top $top_function

            # Add source files and define which optimization to compile using -D flag
            add_files matrix_mult.h -cflags "-Dsolution_${algorithm}_${optimization} -DSIZE=${size}"
            add_files matrix_mult.cpp -cflags "-Dsolution_${algorithm}_${optimization} -DSIZE=${size}"

            # Add testbench files (adjust paths if needed)
            add_files -tb matrix_mult_tb.h -cflags "-Wno-unknown-pragmas"
            add_files -tb matrix_mult_tb.cpp -cflags "-Wno-unknown-pragmas -Dsolution_${algorithm}_${optimization} -DSIZE=${size}"

            # Open optimization for the specific optimization optimization
            open_solution "solution_${algorithm}_${optimization}_${size}"

            # Set the part
            set_part {xc7z020clg484-1} -tool vivado

            # Define clock period and uncertainty
            create_clock -period 10 -name default
            set_clock_uncertainty 1.25

            if {$full_exec} {
                # Run C Simulation (optional)
                csim_design
            }

            # Run Synthesis
            csynth_design

            if {$full_exec} {
                # Run C/RTL Cosimulation
                cosim_design -rtl vhdl

                # Export RTL with custom name based on optimization
                set ip_name "${top_function}_${size}_IP"
                export_design -format ip_catalog -rtl vhdl -ipname ${ip_name}
            }

            # Close the optimization to prepare for the next iteration
            close_solution
        }
    }
}

# Close the project when done
close_project
