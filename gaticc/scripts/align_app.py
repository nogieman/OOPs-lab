from flask import Flask, render_template_string, request
from viz_align import viz_sa_input, viz_fc_bias, viz_sa_pointwise

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_dims = list(map(int, request.form['input_dims'].split()))
        sa_arch = list(map(int, request.form['sa_arch'].split()))
        dram_width = int(request.form['dram_width'])
        visualization_type = request.form['viz_type']
        if visualization_type == 'sa_input':
            table_data = viz_sa_input(input_dims, sa_arch, dram_width)
        elif visualization_type == 'fc_bias':
            vasize = int(request.form['vasize']) if request.form['vasize'] else None
            table_data = viz_fc_bias(input_dims, sa_arch, vasize)
        elif visualization_type == 'sa_pointwise':
            table_data = viz_sa_pointwise(input_dims, sa_arch)
        return render_template_string(template, table_data=table_data,
                                    input_dims=request.form['input_dims'],
                                    sa_arch=request.form['sa_arch'],
                                    dram_width=request.form['dram_width'],
                                    vasize=request.form.get('vasize', ''),
                                    viz_type=visualization_type)
    return render_template_string(template, table_data=None,
                                input_dims='5 3 5', sa_arch='9 8 8',
                                dram_width='32', vasize='', viz_type='sa_input')

template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualization of Aligned Dimensions</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 8px;
            text-align: center;
        }
        input, select {
            padding: 5px;
            margin: 5px;
        }
    </style>
</head>
<body>

<h2>Enter the parameters</h2>
<form method="POST">
    <label for="viz_type">Visualization Type:</label><br>
    <select id="viz_type" name="viz_type" onchange="toggleVasize()">
        <option value="sa_input" {% if viz_type == 'sa_input' %}selected{% endif %}>SA Input</option>
        <option value="fc_bias" {% if viz_type == 'fc_bias' %}selected{% endif %}>FC Bias</option>
        <option value="sa_pointwise" {% if viz_type == 'sa_pointwise' %}selected{% endif %}>SA Pointwise</option>
    </select><br><br>

    <label for="input_dims"> Input Dimensions {% if viz_type == 'sa_input' %} (depth height width): {% else %} (size) {% endif %} </label><br>

    <input type="text" id="input_dims" name="input_dims" value="{{ input_dims }}"><br><br>
    
    <label for="sa_arch">Architecture (depth height width):</label><br>
    <input type="text" id="sa_arch" name="sa_arch" value="{{ sa_arch }}"><br><br>
    
    <label for="dram_width">DRAM Width:</label><br>
    <input type="text" id="dram_width" name="dram_width" value="{{ dram_width }}"><br><br>
    
    <div id="vasize_field" style="display: {% if viz_type == 'fc_bias' %}block{% else %}none{% endif %};">
        <label for="vasize">Vector Array Size:</label><br>
        <input type="text" id="vasize" name="vasize" value="{{ vasize }}"><br><br>
    </div>
    
    <input type="submit" value="Submit">
</form>

{% if table_data %}
    <h3>Output Table:</h3>
    <table>
        <tr>
            <th>Index</th>
            <th>Values</th>
        </tr>
        {% for row in table_data %}
        <tr>
            <td>{{ loop.index }}</td>
            <td>{{ row | join(", ") }}</td>
        </tr>
        {% endfor %}
    </table>
{% endif %}

<script>
function toggleVasize() {
    var vizType = document.getElementById("viz_type").value;
    var vasizeField = document.getElementById("vasize_field");
    vasizeField.style.display = (vizType === "fc_bias") ? "block" : "none";
}
</script>

</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
