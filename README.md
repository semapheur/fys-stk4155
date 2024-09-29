# Julia tips

**Activate project**

From the shell:
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