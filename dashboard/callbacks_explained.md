# Understanding Callbacks in Dash

In Dash, **callbacks** manage interactivity by linking component updates to specific events, such as dropdown changes or button clicks. This guide outlines the steps for defining and using callbacks, with an example demonstrating how to dynamically update options based on user input.

---

## Step-by-Step Overview of Callbacks

### 1. Defining Inputs and Outputs

Each callback specifies:
- **Outputs**: The properties of components (e.g., graph or table) that need updating.
- **Inputs**: The properties of components (e.g., dropdown selections, button clicks) that trigger updates.

### 2. Writing the Callback Function

- Define a function that processes **Inputs** and returns values for **Outputs**.
- This function automatically executes when any **Input** value changes.

### 3. Connecting Inputs and Outputs with `@app.callback`

- Use the **@app.callback** decorator to bind the function to specific components, linking **Inputs** to **Outputs**.
- This setup informs Dash when and where to apply the callback function.

---

## Example: Updating the DOY Selector

This example demonstrates a callback that updates the **DOY (Day of Year) selector** based on the selected folder and year:

```python
@app.callback(
    Output('doy-selector', 'options'),  # Output: Updates options in DOY dropdown
    [Input('folder-selector', 'value'),  # Input: Folder selection dropdown
     Input('year-selector', 'value')]    # Input: Year selection dropdown
)
def update_doy_selector(selected_folder, selected_years):
    if not selected_folder or not selected_years:
        return []  # Return empty options if no selection is made
    # Retrieve DOYs based on folder and year selections
    doys = list_doys(DATA_FOLDER, selected_folder, selected_years)
    # Format DOYs as options for the dropdown
    return [{'label': d, 'value': d} for d in doys]
```

## Explanation:

    - Output: Updates the options property of the doy-selector dropdown based on the function's return value.
    - Inputs: The folder-selector and year-selector dropdown values trigger the callback.
    - Function: update_doy_selector executes each time folder-selector or year-selector changes. If selections are valid, it retrieves the list of DOYs using list_doys and formats them as dropdown options.

## Key Points

    - Stateless: Callbacks only run when triggered and do not retain past data unless stored in a dcc.Store component.
    - Chaining: Multiple callbacks can share inputs and outputs, enabling complex interdependencies across components.
