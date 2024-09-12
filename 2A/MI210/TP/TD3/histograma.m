function bin_counts = histograma(data, array_size, num_bins, min_count, max_count)
    % custom histogram function

    % define histogram bins
    bin_width = (max_count - min_count) / num_bins;
    
    bin_counts = zeros(num_bins, 1);

    for i = 1:array_size
        % calculate which bin spike falls
        bin_index = ceil((data(i) - min_count) / bin_width);

        % correct 0 index
        if bin_index < 1
            bin_index = 1;
        end

        % correct num_bins index
        if bin_index > num_bins
            bin_index = num_bins;
        end

        % increment counter for that bin
        bin_counts(bin_index) = bin_counts(bin_index) + 1;
    end
end