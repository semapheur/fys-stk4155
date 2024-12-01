using DataStructures
using Printf

function dataframe_to_markdown_table(
  df::DataFrame,
  headers::OrderedDict{String,Union{Symbol,Tuple{Symbol,Symbol}}},
)
  # Create the header of the Markdown table
  header_row = "| " * join(keys(headers), " | ") * " |"
  separator_row = "| " * join([":-:" for _ = 1:length(headers)], " | ") * " |"

  # Initialize the Markdown table with the header
  markdown_table = [header_row, separator_row]

  # Iterate over each row in the DataFrame to format it
  for row in eachrow(df)
    markdown_row = "|"

    for (_, column) in headers
      if column isa Symbol
        value = row[column]
        formatted_value = value isa Number ? "\$\\num{$(@sprintf("%.2e", value))}\$" : value

      elseif column isa Tuple{Symbol,Symbol}
        value1, value2 = row[column[1]], row[column[2]]
        formatted_value = "\$\\num{$(@sprintf("%.2e", value1))} \\pm \\num{$(@sprintf("%.2e", value2))}\$"
      end

      markdown_row = markdown_row * " $formatted_value |"
    end

    # Append the formatted row to the Markdown table
    push!(markdown_table, markdown_row)
  end

  # Join all parts into a single string
  return join(markdown_table, "\n")
end