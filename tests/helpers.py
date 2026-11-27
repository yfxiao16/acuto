"""Shared fixtures of the tests: a small tool registry of a refund desk."""

from __future__ import annotations

import acuto


def registry():
    reg = acuto.ToolRegistry()

    @reg.tool
    def lookup_order(order_id: str) -> dict:
        "Look up an order."
        return {"order": order_id}

    @reg.tool
    def issue_refund(order_id: str) -> dict:
        "Refund an order."
        return {"refunded": order_id}

    @reg.tool
    def notify_customer(text: str) -> dict:
        "Email the customer."
        return {"sent": True}

    @reg.tool
    def delete_account() -> dict:
        "Delete the customer's account."
        return {"deleted": True}

    @reg.tool(policy_source=True)
    def get_policy() -> dict:
        "Read the store's refund policy."
        return {"policy": "Once you refund anything you must notify me before finishing."}

    return reg
