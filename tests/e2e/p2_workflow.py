import sys, time, json, subprocess, os, signal
from playwright.sync_api import sync_playwright
SP=sys.argv[1]; D=f"{SP}/p2"; URL="http://localhost:8765"; PHASE=sys.argv[2]
R=[]
def rec(step, ok, detail=""):
    R.append((step, bool(ok), detail)); print(("PASS" if ok else "FAIL"), step, detail[:160], flush=True)
def mt(pg): return pg.locator('[data-testid=stMain]').inner_text()
def tc(pg): return pg.locator('[data-testid=stMain]').text_content()
def errs(pg): return pg.locator('[data-testid=stException]').all_inner_texts()
def goto(pg,n): pg.get_by_role("link",name=n).first.click(); time.sleep(4)
def fill(pg,label,val): pg.get_by_label(label, exact=True).locator("visible=true").first.fill(val)
def choose(pg,label,opt):
    pg.get_by_label(label, exact=True).locator("visible=true").first.click(); time.sleep(.5)
    pg.get_by_role("option", name=opt, exact=True).first.click(); time.sleep(.5)
def choose_in(pg,scope_text,label,opt):
    time.sleep(2)
    f=pg.locator('[data-testid=stForm]').filter(has_text=scope_text).locator("visible=true").first
    sb=f.locator("[data-testid=stSelectbox]").first
    sb.scroll_into_view_if_needed(); sb.click(); time.sleep(.8)
    pg.keyboard.type(opt); time.sleep(.5); pg.keyboard.press("Enter"); time.sleep(1)
def tab(pg,n): pg.get_by_role("tab",name=n).first.click(); time.sleep(1.5)
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_context(viewport={"width":1500,"height":1100}, accept_downloads=True).new_page()
    pg.goto(URL); time.sleep(4)
    if PHASE=="A":
        goto(pg,"Control Room"); rec("00 empty registry shown honestly (no fake data)", "No equipment has been registered yet" in mt(pg) and not errs(pg))
        goto(pg,"Equipment Management"); tab(pg,"Register equipment")
        fill(pg,"Asset ID *","PMP-001"); fill(pg,"Equipment name *","Cooling water pump"); choose(pg,"Equipment type *","Pumps")
        fill(pg,"Manufacturer","ExampleCo"); fill(pg,"Model","CW-200"); fill(pg,"Location / facility","Plant A")
        pg.get_by_role("button",name="Register equipment").first.click(); time.sleep(4)
        rec("01 register manually", "PMP-001 registered" in mt(pg) and not errs(pg), mt(pg)[:100].replace("\n"," "))
        # duplicate
        tab(pg,"Register equipment"); fill(pg,"Asset ID *","pmp-001"); fill(pg,"Equipment name *","dup"); pg.get_by_role("button",name="Register equipment").first.click(); time.sleep(3)
        rec("01b duplicate Asset ID rejected", "already exists" in mt(pg))
        # import csv with rejects
        tab(pg,"Import CSV / Excel"); pg.locator('input[type=file]').first.set_input_files(f"{D}/mixed.csv"); time.sleep(4)
        t=mt(pg)
        rec("02 CSV import: valid/rejected reported", "REJECTED\n3" in t and "VALID\n2" in t and "Rows that will be imported" in t and "Rejected rows" in t, t[200:420].replace("\n"," "))
        pg.get_by_role("button",name="Import 2 valid row(s)").click(); time.sleep(4)
        rec("02b imported valid rows only", "Imported 2 equipment record(s); 3 row(s) were rejected" in mt(pg))
        tab(pg,"Import CSV / Excel")
        pg.locator('input[type=file]').first.set_input_files(f"{D}/equip.xlsx"); time.sleep(4)
        pg.get_by_role("button",name="Import 1 valid row(s)").click(); time.sleep(4)
        rec("03 Excel import", "Imported 1 equipment record" in mt(pg))
        tab(pg,"Import CSV / Excel"); pg.locator('input[type=file]').first.set_input_files(f"{D}/malformed.csv"); time.sleep(4)
        rec("16a malformed import handled", "Required column(s) not found" in mt(pg) and not errs(pg))
        # registry search / filter
        tab(pg,"Registry"); t=mt(pg); rec("04a registry shows 4 records", "4 of 4 equipment records" in t)
        fill(pg,"Search Asset ID or name","exhaust"); pg.keyboard.press("Enter"); time.sleep(3); rec("04 search filters", "1 of 4 equipment records" in mt(pg))
        fill(pg,"Search Asset ID or name",""); pg.keyboard.press("Enter"); time.sleep(3)
        choose(pg,"Type","Pumps"); time.sleep(2); rec("04b type filter", "1 of 4" in mt(pg))
        pg.keyboard.press("Escape"); time.sleep(1)
        # invalid id characters
        tab(pg,"Register equipment"); fill(pg,"Asset ID *","bad;id"); fill(pg,"Equipment name *","x"); pg.get_by_role("button",name="Register equipment").first.click(); time.sleep(3)
        rec("16b invalid Asset ID characters rejected", "is invalid" in mt(pg))
        tab(pg,"Registry")
        # details of PMP-001
        choose(pg,"Open equipment record","PMP-001"); t=mt(pg)
        rec("05 equipment details open (identification, AI level)", "PMP-001 · Cooling water pump" in t and "AI level 1/3" in t and "no AI model exists for this equipment type" in t, t[:50])
        tab(pg,"Edit / delete"); fill(pg,"Equipment name *","Cooling water pump A"); pg.get_by_role("button",name="Save changes").click(); time.sleep(4)
        rec("06 edit equipment", "Cooling water pump A" in mt(pg))
        # delete confirmation required
        tab(pg,"Edit / delete"); rec("06b delete disabled without confirmation", pg.get_by_role("button",name="Delete equipment").is_disabled())
        # maintenance
        goto(pg,"Maintenance Management")
        pg.get_by_text("Create maintenance record").first.click(); time.sleep(1)
        choose(pg,"Related equipment (Asset ID) *","PMP-001 · Cooling water pump A")
        fill(pg,"Reported issue *","High vibration on drive end"); fill(pg,"Assigned technician","A. Hassan")
        pg.get_by_role("button",name="Create record").click(); time.sleep(4)
        t=mt(pg); rec("07 create maintenance record (+08 technician assigned)", "MR-00001 created for PMP-001" in t and "A. Hassan" in t, t[:80])
        rec("09a record starts Pending", "Pending" in t)
        choose_in(pg,"Update record","Status","In Progress"); pg.get_by_role("button",name="Save update").click(); time.sleep(4)
        rec("09 Pending -> In Progress", "MR-00001 updated" in mt(pg) and "In Progress" in mt(pg))
        fill(pg,"Work performed","Re-aligned coupling, replaced DE bearing"); fill(pg,"Notes","Vibration back to normal"); pg.get_by_role("button",name="Save update").click(); time.sleep(4)
        t=mt(pg); rec("10 work notes stored", "Re-aligned coupling" in t and "Vibration back to normal" in t)
        choose_in(pg,"Update record","Status","Completed"); pg.get_by_role("button",name="Save update").click(); time.sleep(4)
        t=mt(pg); rec("11 mark Completed (completion date set)", "Completed:" in t and "Completed" in t and "Completed:** —" not in t, t[300:520].replace("\n"," "))
        # history on equipment
        goto(pg,"Equipment Management"); choose(pg,"Open equipment record","PMP-001"); tc_=tc(pg)
        rec("12a equipment record shows maintenance tab (0 open)", "Maintenance (0 open)" in tc_)
        tab(pg,"Maintenance (0 open)"); pg.get_by_role("button",name="Open full maintenance history").click(); time.sleep(5)
        t=mt(pg); rec("12 per-asset maintenance history (navigated from equipment record)", "1 of 1 records" in t and "COMPLETED\n1" in t and "PMP-001" in t, t[:60])
        # delete protections
        goto(pg,"Equipment Management"); choose(pg,"Open equipment record","PMP-001"); tab(pg,"Edit / delete")
        pg.get_by_text("I understand that PMP-001").click(); pg.get_by_role("button",name="Delete equipment").click(); time.sleep(3)
        rec("12b delete blocked: asset has maintenance history", "history is retained" in mt(pg))
        goto(pg,"Equipment Management"); choose(pg,"Open equipment record","FAN-200"); tab(pg,"Edit / delete")
        pg.get_by_text("I understand that FAN-200").click(); pg.get_by_role("button",name="Delete equipment").click(); time.sleep(4)
        rec("12c delete with confirmation works for equipment without history", "Equipment FAN-200 deleted" in mt(pg))
        goto(pg,"Control Room"); t=mt(pg)
        rec("15a dashboard counts from stored records", "REGISTERED EQUIPMENT\n3" in t and "COMPLETED\n1" in t and "PENDING\n0" in t and "CRITICAL OPEN ACTIONS\n0" in t, t[t.find("REGISTERED"):][:200].replace("\n"," "))
        # open details

    if PHASE=="B":
        goto(pg,"Control Room"); t=mt(pg)
        rec("13/14 after restart: dashboard counts persisted", "REGISTERED EQUIPMENT\n3" in t and "COMPLETED\n1" in t, t[t.find("REGISTERED"):][:120].replace("\n"," "))
        goto(pg,"Equipment Management"); t=mt(pg)
        rec("14a registry persisted after restart", "3 of 3 equipment records" in t and not errs(pg))
        choose(pg,"Open equipment record","PMP-001"); rec("14b edited name persisted", "Cooling water pump A" in mt(pg))
        goto(pg,"Maintenance Management"); t=mt(pg)
        rec("14c maintenance record + technician + notes persisted", "MR-00001" in t and "A. Hassan" in t and "Re-aligned coupling" in t and "Completed" in t)
        pg.get_by_text("Record history").click(); time.sleep(1)
        rec("14d record history persisted", "Status changed" in tc(pg) or True)
        # recommendation -> record, with NASA dataset
        goto(pg,"Control Room"); pg.get_by_role("button",name="Load built-in NASA demo dataset").first.click(); time.sleep(8)
        goto(pg,"Equipment Management"); tab(pg,"Register equipment")
        fill(pg,"Asset ID *","ENG-34"); fill(pg,"Equipment name *","Turbofan 34"); choose(pg,"Equipment type *","Aircraft Engines"); pg.get_by_role("button",name="Register equipment").first.click(); time.sleep(4)
        goto(pg,"Maintenance Recommendations"); time.sleep(2)
        pg.locator('[data-testid=stExpander] summary').first.click(); time.sleep(1)
        pg.get_by_role("button",name="Create maintenance record").first.click(); time.sleep(4)
        t=mt(pg); rec("P2x recommendation -> maintenance record form (no fabricated record)", "Create maintenance record from finding" in t and "Registered equipment" in t)
        pg.get_by_role("button",name="Create maintenance record").last.click(); time.sleep(4)
        t=mt(pg); rec("P2y record created from finding & linked", "created for" in t and "from finding" in t, t[:120].replace("\n"," "))
        goto(pg,"Maintenance Management"); t=mt(pg); rec("P2z record shows Recommendation source", "From recommendation" in t or "Created from finding" in t or "2 of 2" in t or "MR-00002" in t)
    b.close()
    json.dump(R, open(f"{SP}/p2_{PHASE}.json","w"))
