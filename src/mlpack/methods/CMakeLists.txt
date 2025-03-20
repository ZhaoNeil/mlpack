# Utility macro to add bindings for all languages.
macro(add_all_bindings directory name type)
  add_category(${name} ${type})
  add_cli_executable(${directory} ${name})
  add_python_binding(${directory} ${name})
  add_julia_binding(${directory} ${name})
  add_go_binding(${directory} ${name})
  add_r_binding(${directory} ${name})
  add_markdown_docs(${directory} ${name} "cli;python;julia;go;r" "")
endmacro()

# First, define all the "regular" bindings that exist for all languages.
add_all_bindings(reinforcement_learning reinforcement_learning "Misc. / Other")

# Now, define the "special" bindings that are different somehow.

# Adaboost has a Python wrapper class that encapsulates its functionality in a
# few methods.
add_category(adaboost "Classification")
group_bindings(adaboost adaboost "train;classify;probabilities")
add_python_wrapper(adaboost adaboost)

add_cli_executable(adaboost adaboost)
add_julia_binding(adaboost adaboost)
add_go_binding(adaboost adaboost)
add_r_binding(adaboost adaboost)
add_markdown_docs(adaboost adaboost "cli;python;julia;go;r" "python")

# Linear Regression has a Python wrapper class that encapsulates its
# functionality in a few methods.
add_category(linear_regression "Regression")
group_bindings(linear_regression linear_regression "train;predict")
add_python_wrapper(linear_regression linear_regression)

add_cli_executable(linear_regression linear_regression)
add_julia_binding(linear_regression linear_regression)
add_go_binding(linear_regression linear_regression)
add_r_binding(linear_regression linear_regression)
add_markdown_docs(linear_regression linear_regression "cli;python;julia;go;r"
    "python")

# The imputer is only defined for CLI bindings right now.
add_category(preprocess_imputer "Preprocessing")
add_cli_executable(preprocess preprocess_imputer)
add_markdown_docs(preprocess preprocess_imputer "cli" "")

# The image converter is only enabled if STB is available.
if (STB_AVAILABLE)
  add_all_bindings(preprocess image_converter "Preprocessing")
endif()

# Range search provides a vector of vector of results, and this is only
# supported for the CLI bindings.
add_category(range_search "Geometry")
add_cli_executable(range_search range_search)
add_markdown_docs(range_search range_search "cli" "")
