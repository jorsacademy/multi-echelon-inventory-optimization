from multi_echelon_inventory import InventoryNode, PolicyType
from multi_echelon_inventory.policies import policy_targets


def nodes():
    return [
        InventoryNode("Upstream", 100, 20, 4, 4, 1.0, 9.0, service_level=0.95),
        InventoryNode("Downstream", 80, 10, 3, 2, 1.0, 9.0, service_level=0.95),
    ]


def test_all_policy_targets_are_non_negative() -> None:
    for policy in PolicyType:
        targets = policy_targets(nodes(), policy)
        assert all(value >= 0 for value in targets.values())


def test_echelon_upstream_target_exceeds_downstream_target() -> None:
    targets = policy_targets(nodes(), PolicyType.ECHELON_BASE_STOCK)
    assert targets["Upstream"] > targets["Downstream"]


def test_critical_ratio_target_is_finite() -> None:
    target = policy_targets(nodes(), PolicyType.CRITICAL_RATIO)["Upstream"]
    assert target > 0
