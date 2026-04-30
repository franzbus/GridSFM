#!/usr/bin/env julia
# patch_model.jl — Add missing required keys to a PowerModels-format JSON file.
#
# The GridSFM data-release models were exported before PowerModels.jl v0.21+
# added mandatory "storage" and "switch" component dicts. This script patches
# a model JSON in-place (or writes it to an output path) by adding empty
# dicts for any missing required components.
#
# Usage:
#   julia patch_model.jl <input.json> [output.json]
#
# If output is omitted, the patched file is written to the same path as input.

using JSON

if length(ARGS) < 1
    println("Usage: julia patch_model.jl <input.json> [output.json]")
    exit(1)
end

input_path = ARGS[1]
output_path = length(ARGS) >= 2 ? ARGS[2] : input_path

if !isfile(input_path)
    println("Error: file not found: $input_path")
    exit(1)
end

data = JSON.parsefile(input_path)

# Required component dicts that PowerModels.jl expects
patched = String[]
for key in ["storage", "switch"]
    if !haskey(data, key)
        data[key] = Dict{String,Any}()
        push!(patched, key)
    end
end

open(output_path, "w") do io
    JSON.print(io, data)
end

if isempty(patched)
    println("  $(basename(input_path)) → $(basename(output_path)): no patching needed")
else
    println("  $(basename(input_path)) → $(basename(output_path)): added $(join(patched, ", "))")
end
