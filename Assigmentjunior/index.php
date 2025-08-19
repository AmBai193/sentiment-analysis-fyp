<?php
function calculateRates($voltage, $current, $rate) {
    $power = $voltage * $current; // in Watts
    $results = [];

    for ($hour = 1; $hour <= 24; $hour++) {
        $energy = ($power * $hour) / 1000; // kWh
        $total = $energy * ($rate / 100);  // RM
        $results[] = [
            "hour" => $hour,
            "energy" => round($energy, 5),
            "total" => round($total, 2)
        ];
    }

    return [
        "power_kw" => round($power / 1000, 5),
        "rate_rm" => round($rate / 100, 3),
        "data" => $results
    ];
}

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $voltage = floatval($_POST['voltage']);
    $current = floatval($_POST['current']);
    $rate = floatval($_POST['rate']);

    $calculation = calculateRates($voltage, $current, $rate);
}
?>
<!DOCTYPE html>
<html>
<head>
    <title>Electricity Calculator</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.0/dist/css/bootstrap.min.css">
</head>
<body class="p-4">
<div class="container">
    <h2 class="mb-4">Electricity Consumption Calculator</h2>
    <form method="post" class="mb-4">
        <div class="form-group">
            <label>Voltage (V)</label>
            <input type="number" step="0.01" name="voltage" class="form-control" required>
        </div>
        <div class="form-group">
            <label>Current (A)</label>
            <input type="number" step="0.01" name="current" class="form-control" required>
        </div>
        <div class="form-group">
            <label>Rate (sen/kWh)</label>
            <input type="number" step="0.01" name="rate" class="form-control" required>
        </div>
        <button type="submit" class="btn btn-primary">Calculate</button>
    </form>

    <?php if (!empty($calculation)): ?>
        <h4>Results</h4>
        <p><strong>Power:</strong> <?= $calculation['power_kw'] ?> kW</p>
        <p><strong>Rate:</strong> <?= $calculation['rate_rm'] ?> RM</p>

        <table class="table table-bordered">
            <thead>
                <tr>
                    <th># Hour</th>
                    <th>Energy (kWh)</th>
                    <th>Total (RM)</th>
                </tr>
            </thead>
            <tbody>
                <?php foreach ($calculation['data'] as $row): ?>
                    <tr>
                        <td><?= $row['hour'] ?></td>
                        <td><?= $row['energy'] ?></td>
                        <td><?= number_format($row['total'], 2) ?></td>
                    </tr>
                <?php endforeach; ?>
            </tbody>
        </table>
    <?php endif; ?>
</div>
</body>
</html>
<?php