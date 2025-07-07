# Julia tips

**Activate project**

In the shell:
```shell
julia --project
```

In Julia:
```jl
using Pkg
Pkg.activate(".")
```

**Activate and install necessary Julia Packages**

Ensure that the project is activated and run

```jl
using Pkg
Pkg.instantiate()
```

# Python tips

**Install dependencies using uv**

In the shell:
```shell
uv run
```