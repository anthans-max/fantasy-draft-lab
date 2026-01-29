def calculate_fantasy_points(
    rushing_yards,
    receiving_yards,
    receptions,
    touchdowns,
    scoring_format="ppr"
):
    points = (
        rushing_yards * 0.1
        + receiving_yards * 0.1
        + touchdowns * 6
    )

    if scoring_format == "ppr":
        points += receptions * 1.0
    elif scoring_format == "half_ppr":
        points += receptions * 0.5

    return points
