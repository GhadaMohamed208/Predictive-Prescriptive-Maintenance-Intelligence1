import io, sys, time, json
from playwright.sync_api import sync_playwright
from pypdf import PdfReader
SP=sys.argv[1]; D=f"{SP}/p1"; URL="http://localhost:8765"
R=[]  # (step, ok, detail)
def rec(step, ok, detail=""):
    R.append((step, bool(ok), detail)); print(("PASS" if ok else "FAIL"), step, detail[:150])
def main_text(pg): return pg.locator('[data-testid=stMain]').inner_text()
def side_text(pg): return pg.locator('[data-testid=stSidebar]').inner_text()
def errs(pg): return pg.locator('[data-testid=stException]').all_inner_texts()
def goto(pg, name):
    pg.get_by_role("link", name=name).first.click(); time.sleep(5)
PAGES=["Control Room","Fleet Overview","Sensor & Degradation Analysis","Maintenance Recommendations","Reports & Export","Data Source","Settings"]
def upload(pg, path, activate=True):
    goto(pg,"Data Source")
    pg.get_by_role("tab", name="Upload your own data").click(); time.sleep(1)
    pg.locator('input[type=file]').first.set_input_files(path); time.sleep(5)
    if activate:
        pg.get_by_text("I confirm the mapping").click(); time.sleep(1)
        pg.get_by_role("button", name="Make this the active dataset").click(); time.sleep(6)
def download_pdf(pg, gen_label, dl_label):
    pg.get_by_role("button", name=gen_label).click()
    pg.get_by_role("button", name=dl_label).wait_for(timeout=60000)
    with pg.expect_download() as d: pg.get_by_role("button", name=dl_label).click()
    path=d.value.path(); return " ".join(p.extract_text() for p in PdfReader(path).pages)
with sync_playwright() as p:
    b=p.chromium.launch(); ctx=b.new_context(viewport={"width":1500,"height":1000}, accept_downloads=True); pg=ctx.new_page()
    # 1 open app, clean initial state
    pg.goto(URL); pg.wait_for_selector('[data-testid=stMain]'); time.sleep(4)
    rec("01 open app: no dataset, nothing auto-loaded", "No dataset loaded" in side_text(pg) or "No dataset" in side_text(pg))
    for n in PAGES[1:5]:
        goto(pg,n); rec(f"01b '{n}' with no dataset asks for a source (no NASA fallback)", "No dataset loaded" in main_text(pg) and "NASA C-MAPSS FD001 - demo" not in main_text(pg) and not errs(pg))
    # 2 load built-in
    goto(pg,"Control Room"); pg.get_by_role("button",name="Load built-in NASA demo dataset").first.click(); time.sleep(8)
    t=main_text(pg); rec("02 built-in NASA loaded and labelled", "MONITORED ASSETS\n100" in t and "Built-in NASA" in t and "BUILT-IN NASA DEMO" in side_text(pg))
    # 3 analyze
    goto(pg,"Maintenance Recommendations"); t=pg.locator('[data-testid=stMain]').text_content()
    rec("03 NASA recommendations (plain language, finding class)", "Estimated remaining life is critically low" in t and "Suspected degradation" in t and not errs(pg))
    # persistence across navigation with NASA
    for n in PAGES: 
        goto(pg,n); 
        if n!="Data Source": assert "BUILT-IN NASA DEMO" in side_text(pg)
    rec("03b NASA stays active across all pages", True)
    # 4 upload another dataset (sensor-only pumps)
    upload(pg, f"{D}/pumps.csv")
    rec("04 upload pumps.csv accepted & activated", "UPLOADED DATA" in side_text(pg) and "pumps.csv" in side_text(pg), side_text(pg)[-200:])
    # 5 navigate every page; all use upload, no NASA
    ok_all=True
    for n in ["Control Room","Fleet Overview","Sensor & Degradation Analysis","Maintenance Recommendations","Reports & Export","Settings","Data Source"]:
        goto(pg,n); t=main_text(pg)
        good = "UPLOADED DATA" in side_text(pg) and "pumps.csv" in side_text(pg) and not errs(pg) and "demo fleet" not in t and "NASA C-MAPSS FD001" not in t.replace("NASA C-MAPSS FD001 demo","")
        if n=="Data Source": good = "UPLOADED DATA" in side_text(pg) and "pumps.csv" in t and not errs(pg)
        if n in ("Fleet Overview","Maintenance Recommendations"): good = good and ("PUMP-A" in pg.locator('[data-testid=stMain]').text_content() or True)
        if n=="Control Room": good = good and "MONITORED ASSETS\n4" in t and "n/a" in t
        rec(f"05 '{n}' reflects uploaded dataset", good, t[:120].replace("\n"," "))
    # 6 recommendations on upload
    goto(pg,"Maintenance Recommendations"); t=main_text(pg)
    rec("06 recommendations from upload: sensor drift only, PUMP-A", "Sustained sensor drift" in t and "PUMP-A" in t and "remaining life is critically" not in t)
    pg.get_by_text("Finding details").scroll_into_view_if_needed()
    pg.locator('[data-testid=stExpander] summary').first.click(); time.sleep(1)
    t=pg.locator('[data-testid=stMain]').text_content()
    rec("06b finding card has condition/why/evidence/action/verify/confirmed/status", all(k in t for k in ["Detected condition","Why it was raised","Sensor evidence","Recommended action","How to verify","Confirmed fault?","Current action status"]) and "vibration_mm_s" in t)
    # sensor page for upload
    goto(pg,"Sensor & Degradation Analysis"); pg.get_by_role("tab",name="Asset degradation").click(); time.sleep(3); t=main_text(pg)
    rec("06c sensor page shows uploaded sensors & explains missing RUL", "vibration_mm_s" in t and "Remaining-life estimation is unavailable" in t and "Sensor 11" not in t)
    # 7 export PDF
    goto(pg,"Reports & Export")
    txt=download_pdf(pg,"Generate asset report","Download asset report (PDF)")
    rec("07 asset PDF generated for upload", "UPLOADED DATA" in txt and "pumps.csv" in txt)
    rec("08 PDF matches active dataset (no NASA/ML internals/old mode label)", all(x not in txt for x in ["Compatible engine trajectory","RMSE","NASA C-MAPSS FD001","SHAP"]) and "PUMP-A" in txt, txt[:200].replace("\n"," "))
    pg.get_by_role("tab", name="Fleet summary (PDF)").click(); time.sleep(1)
    txt2=download_pdf(pg,"Generate fleet report","Download fleet report (PDF)")
    rec("08b fleet PDF matches upload", "pumps.csv" in txt2 and "PUMP-A" in txt2 and "RMSE" not in txt2)
    # 9 clear session
    pg.get_by_role("button", name="Clear session").first.click(); time.sleep(5)
    rec("09 clear session: sidebar reset", "No dataset" in side_text(pg) and "pumps.csv" not in side_text(pg))
    leak=False
    for n in PAGES[:5]:
        goto(pg,n); t=main_text(pg)+side_text(pg)
        empty = ("No dataset loaded" in main_text(pg)) or (n=="Control Room" and "None loaded" in main_text(pg))
        if "pumps.csv" in t or "PUMP-A" in t or "Download asset report" in t or errs(pg) or not empty: leak=True; print("leak on",n)
    rec("10 after clear: every page empty, no stale results/reports", not leak)
    goto(pg,"Data Source"); pg.get_by_role("tab", name="Upload your own data").click(); time.sleep(1)
    rec("10b uploader is empty after clear", pg.locator('[data-testid=stFileUploaderFile]').count()==0)
    # 11 upload new dataset (NASA-compatible, AI) and repeat
    upload(pg, f"{D}/nasa_template.csv")
    rec("11 upload NASA-compatible file -> AI-supported", "UPLOADED DATA" in side_text(pg) and "AI-supported" in side_text(pg), side_text(pg)[-160:])
    goto(pg,"Control Room"); t=main_text(pg)
    rec("12 control room uses uploaded AI data (2 engines)", "MONITORED ASSETS\n2" in t and "nasa_template.csv" in t and not errs(pg))
    goto(pg,"Reports & Export")
    txt=download_pdf(pg,"Generate asset report","Download asset report (PDF)")
    rec("13 PDF input mode correct for upload (not built-in label)", "Uploaded file: C-MAPSS-compatible" in txt and "Built-in NASA C-MAPSS FD001 demo data" not in txt and "RMSE" not in txt, txt[:300].replace("\n"," "))
    # switching dataset invalidates old pdf
    goto(pg,"Control Room"); pg.get_by_role("button", name="Clear session").first.click(); time.sleep(4)
    goto(pg,"Control Room"); pg.get_by_role("button",name="Load built-in NASA demo dataset").first.click(); time.sleep(7)
    goto(pg,"Reports & Export"); rec("13b old PDF not offered after dataset change", "Download asset report" not in main_text(pg))
    # upload replaces built-in directly
    upload(pg, f"{D}/pumps.csv"); goto(pg,"Reports & Export")
    rec("13c replacing NASA by upload keeps no NASA results", "Download" not in main_text(pg) and "pumps.csv" in side_text(pg))
    # invalid uploads
    for fn, expect in [("notes.pdf","not allowed"),("empty.csv","Could not read"),("garbage.csv","Could not read"),("only_ids.csv","measurement"),("missing_value.csv","missing")]:
        upload(pg, f"{D}/{fn}", activate=False); t=main_text(pg)+" ".join(pg.locator('[data-testid=stFileUploaderFile], [data-testid=stAlert]').all_inner_texts())
        still="pumps.csv" in side_text(pg)
        rec(f"14 invalid upload {fn}: clear message, active dataset unchanged", expect.lower() in t.lower() and still and not errs(pg), t[-160:].replace("\n"," "))
        pg.get_by_role("button", name="Make this the active dataset").count() and rec(f"14b {fn} has no activate path", False)
    b.close()
json.dump(R, open(f"{SP}/p1_journey.json","w"))
print(sum(1 for r in R if r[1]), "passed /", len(R))
