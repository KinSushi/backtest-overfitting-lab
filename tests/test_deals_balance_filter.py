"""Non-regression: an MT5 export with a deposit (DEAL_TYPE_BALANCE) must NOT pollute the trades."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_mt5_deals as D

HEADER = "Time,Deal,Position,Symbol,Type,Direction,Volume,Price,Commission,Swap,Fee,Profit"

def _csv(rows):
    return "\n".join([HEADER] + rows)

def test_balance_deposit_is_filtered():
    rows = [
        "2024.01.01 00:00:00,1,0,,2,0,0.00,0.00,0.00,0.00,0.00,10000.00",   # DÉPÔT (type 2)
        "2024.01.02 10:00:00,2,1001,XAUUSD,0,0,0.10,2000.00,-0.70,0,0,0.00", # entry
        "2024.01.02 12:00:00,3,1001,XAUUSD,0,1,0.10,2010.00,-0.70,0,0,100.00",  # exit +100
    ]
    deals = D.parse_deals_csv(_csv(rows))
    # the deposit is removed -> 2 deals (1 trade)
    assert len(deals) == 2
    trades = D.trades_from_deals(deals)
    assert len(trades) == 1
    # net = 100 - 0.70 - 0.70 = 98.6 (NOT 10098.6)
    assert abs(trades[0]["net_pnl"] - 98.6) < 1e-6

def test_credit_and_charge_types_filtered():
    rows = [
        "2024.01.01 00:00:00,1,0,,2,0,0,0,0,0,0,5000.00",     # balance
        "2024.01.01 00:00:00,2,0,,3,0,0,0,0,0,0,500.00",      # credit (type 3)
        "2024.01.02 10:00:00,3,2001,XAUUSD,1,0,0.10,2000,-0.70,0,0,0.00",
        "2024.01.02 12:00:00,4,2001,XAUUSD,1,1,0.10,1990,-0.70,0,0,80.00",
    ]
    deals = D.parse_deals_csv(_csv(rows))
    assert len(deals) == 2                       # only the 2 trade deals remain
    assert all(d["type"] in (0, 1) for d in deals)

def test_filter_can_be_disabled():
    rows = ["2024.01.01 00:00:00,1,0,,2,0,0,0,0,0,0,10000.00"]
    assert len(D.parse_deals_csv(_csv(rows), drop_non_trades=False)) == 1   # kept if disabled
    assert len(D.parse_deals_csv(_csv(rows))) == 0                          # filtered by default

def test_fallback_when_type_column_missing():
    # no Type column -> fallback on Position==0 to spot the deposit
    hdr = "Time,Deal,Position,Symbol,Direction,Volume,Price,Commission,Swap,Fee,Profit"
    rows = [
        hdr,
        "2024.01.01 00:00:00,1,0,,0,0,0,0,0,0,10000.00",                    # Position 0 -> deposit
        "2024.01.02 10:00:00,2,3001,XAUUSD,0,0.10,2000,-0.70,0,0,0.00",
        "2024.01.02 12:00:00,3,3001,XAUUSD,1,0.10,2010,-0.70,0,0,100.00",
    ]
    deals = D.parse_deals_csv("\n".join(rows))
    assert len(deals) == 2                       # the deposit (Position 0) is removed
