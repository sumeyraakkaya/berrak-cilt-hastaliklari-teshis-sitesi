const SINIFLAR = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
const ADLAR = {
  akiec: "Aktinik keratoz (kanser öncesi olabilir)",
  bcc: "Bazal hücreli karsinom (cilt kanseri türü)",
  bkl: "İyi huylu keratoz benzeri leke",
  df: "Dermatofibrom (iyi huylu nodül)",
  mel: "Melanom (en tehlikeli cilt kanseri türü)",
  nv: "Ben (nevüs, genelde iyi huylu)",
  vasc: "Damarsal lezyon"
};
const DIKKAT = ["mel", "bcc", "akiec"];
const $ = (s) => document.querySelector(s);
let model = null;
let objeUrl = null;
let sonOzet = "";

/* ---------- Model yükleme ---------- */
async function modeliYukle() {
  const d = $("#model-durum");
  try {
    if (typeof tf === "undefined") throw new Error("TensorFlow.js yüklenemedi (internet bağlantısı?)");
    model = await tf.loadLayersModel("model/model.json");
    d.textContent = "Yapay zeka modeli hazır. Fotoğraf hem modelle hem otomatik ABCD analiziyle incelenir.";
  } catch (e) {
    console.error("Model yüklenemedi:", e);
    d.textContent = "Yapay zeka modeli yüklenemedi (ayrıntı: F12 > Console). Otomatik ABCD analizi ve belirtiler yine çalışır.";
  }
}

/* ---------- Fotoğraf seçme ---------- */
function fotoAyarla(dosya) {
  if (!dosya || !dosya.type.startsWith("image/")) return;
  if (objeUrl) URL.revokeObjectURL(objeUrl);
  objeUrl = URL.createObjectURL(dosya);
  const img = $("#onizleme");
  img.src = objeUrl;
  img.hidden = false;
  $("#foto-sil").hidden = false;
}
function fotoSil() {
  const img = $("#onizleme");
  img.hidden = true;
  img.removeAttribute("src");
  $("#foto").value = "";
  $("#kamera").value = "";
  $("#foto-sil").hidden = true;
}
$("#foto").addEventListener("change", (e) => fotoAyarla(e.target.files[0]));
$("#kamera").addEventListener("change", (e) => fotoAyarla(e.target.files[0]));
$("#kamera-dugme").addEventListener("click", () => $("#kamera").click());
$("#foto-sil").addEventListener("click", fotoSil);
const alan = $("#alan");
["dragenter", "dragover"].forEach((o) =>
  alan.addEventListener(o, (e) => { e.preventDefault(); alan.classList.add("surukle"); }));
["dragleave", "drop"].forEach((o) =>
  alan.addEventListener(o, (e) => { e.preventDefault(); alan.classList.remove("surukle"); }));
alan.addEventListener("drop", (e) => fotoAyarla(e.dataTransfer.files[0]));

/* ---------- Model tahmini ---------- */
async function tahminEt() {
  const img = $("#onizleme");
  await img.decode();
  const x = tf.tidy(() =>
    tf.browser.fromPixels(img).resizeBilinear([160, 160])
      .toFloat().div(127.5).sub(1).expandDims(0));
  const cikti = model.predict(x);
  const p = Array.from(await cikti.data());
  x.dispose();
  cikti.dispose();
  return p;
}

/*ANALIZ-BASLA*/
function lekeAnaliz(d, N) {
  const medyan = (a) => { a.sort((x, y) => x - y); return a[a.length >> 1]; };
  const R = [], G = [], B = [], E = 6;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    if (x < E || y < E || x >= N - E || y >= N - E) {
      const i = (y * N + x) * 4;
      R.push(d[i]); G.push(d[i + 1]); B.push(d[i + 2]);
    }
  }
  const bg = [medyan(R), medyan(G), medyan(B)];
  const tum = N * N, hist = new Array(256).fill(0), dist = new Float32Array(tum);
  for (let p = 0; p < tum; p++) {
    const i = p * 4, dr = d[i] - bg[0], dg = d[i + 1] - bg[1], db = d[i + 2] - bg[2];
    const v = Math.min(255, Math.sqrt(dr * dr + dg * dg + db * db) * 0.58);
    dist[p] = v; hist[Math.round(v)]++;
  }
  let toplam = 0; for (let i = 0; i < 256; i++) toplam += i * hist[i];
  let wB = 0, sB = 0, maks = 0, esik = 30;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) continue;
    const wF = tum - wB; if (!wF) break;
    sB += t * hist[t];
    const mB = sB / wB, mF = (toplam - sB) / wF, v = wB * wF * (mB - mF) * (mB - mF);
    if (v > maks) { maks = v; esik = t; }
  }
  esik = Math.max(esik, 22);
  let m = new Uint8Array(tum);
  for (let p = 0; p < tum; p++) m[p] = dist[p] > esik ? 1 : 0;
  for (let gecis = 0; gecis < 2; gecis++) {
    const y2 = new Uint8Array(tum);
    for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
      let s = 0;
      for (let j = -1; j <= 1; j++) for (let k = -1; k <= 1; k++) s += m[(y + j) * N + x + k];
      y2[y * N + x] = s >= 5 ? 1 : 0;
    }
    m = y2;
  }
  const et = new Int32Array(tum), yigin = [];
  let no = 0, enBoy = 0, enNo = 0;
  for (let p = 0; p < tum; p++) {
    if (!m[p] || et[p]) continue;
    no++; let boy = 0; yigin.push(p); et[p] = no;
    while (yigin.length) {
      const q = yigin.pop(); boy++;
      const x = q % N, y = (q / N) | 0;
      const komsu = [x > 0 ? q - 1 : -1, x < N - 1 ? q + 1 : -1, y > 0 ? q - N : -1, y < N - 1 ? q + N : -1];
      for (const r of komsu) if (r >= 0 && m[r] && !et[r]) { et[r] = no; yigin.push(r); }
    }
    if (boy > enBoy) { enBoy = boy; enNo = no; }
  }
  const alan = enBoy, oran = alan / tum;
  if (oran < 0.015 || oran > 0.7) return null;
  const mk = new Uint8Array(tum);
  let cx = 0, cy = 0;
  for (let p = 0; p < tum; p++) if (et[p] === enNo) { mk[p] = 1; cx += p % N; cy += (p / N) | 0; }
  cx /= alan; cy /= alan;
  let yatay = 0, dikey = 0, sinir = 0, sr = 0, sg = 0, sb = 0, sr2 = 0, sg2 = 0, sb2 = 0;
  const sinirlar = [];
  for (let p = 0; p < tum; p++) {
    if (!mk[p]) continue;
    const x = p % N, y = (p / N) | 0;
    const mx = Math.round(2 * cx - x), my = Math.round(2 * cy - y);
    if (mx < 0 || mx >= N || !mk[y * N + mx]) yatay++;
    if (my < 0 || my >= N || !mk[my * N + x]) dikey++;
    if (x === 0 || y === 0 || x === N - 1 || y === N - 1 || !mk[p - 1] || !mk[p + 1] || !mk[p - N] || !mk[p + N]) { sinir++; sinirlar.push(p); }
    const i = p * 4;
    sr += d[i]; sg += d[i + 1]; sb += d[i + 2];
    sr2 += d[i] * d[i]; sg2 += d[i + 1] * d[i + 1]; sb2 += d[i + 2] * d[i + 2];
  }
  const sap = (a, a2) => Math.sqrt(Math.max(0, a2 / alan - (a / alan) * (a / alan)));
  const sd = (sap(sr, sr2) + sap(sg, sg2) + sap(sb, sb2)) / 3;
  const asim = (yatay + dikey) / (2 * alan);
  const duzensiz = sinir / (2 * Math.sqrt(Math.PI * alan));
  const kes = (v) => Math.max(0, Math.min(100, Math.round(v * 100)));
  return {
    asim: kes((asim - 0.04) / 0.36),
    kenar: kes((duzensiz - 1.0) / 0.7),
    renk: kes((sd - 8) / 28),
    ham: { asim, duzensiz, sd, oran },
    maske: mk, sinirlar
  };
}
/*ANALIZ-BITIS*/

/* Fotoğrafı 128x128 kareye alır, lekeyi bulur, çerçevesini tuvale çizer */
async function fotoAnalizEt() {
  const img = $("#onizleme");
  await img.decode();
  const N = 128, w = img.naturalWidth, h = img.naturalHeight, k = Math.min(w, h);
  const c = document.createElement("canvas");
  c.width = c.height = N;
  const g = c.getContext("2d", { willReadFrequently: true });
  g.drawImage(img, (w - k) / 2, (h - k) / 2, k, k, 0, 0, N, N);
  const sonuc = lekeAnaliz(g.getImageData(0, 0, N, N).data, N);
  const tuval = $("#oto-tuval"), t = tuval.getContext("2d");
  t.drawImage(img, (w - k) / 2, (h - k) / 2, k, k, 0, 0, tuval.width, tuval.height);
  if (sonuc) {
    const s = tuval.width / N;
    t.fillStyle = "#8ec5f0";
    sonuc.sinirlar.forEach((p) => t.fillRect((p % N) * s - 1, ((p / N) | 0) * s - 1, s + 2, s + 2));
  }
  return sonuc;
}

/* ---------- Risk seviyesi ---------- */
function seviyeHesapla(sayi, dikkat) {
  if (dikkat >= 0.5 || sayi >= 4 || (dikkat >= 0.3 && sayi >= 2)) return "yuksek";
  if (dikkat >= 0.25 || sayi >= 2) return "orta";
  return "dusuk";
}
const METIN = {
  dusuk: ["Düşük düzeyde uyarı", "Belirgin bir uyarı işareti saptanmadı. Yine de leke büyür, renk değiştirir ya da kanarsa bir hekime görün."],
  orta: ["Takip önerilir", "Bazı uyarı işaretleri var. Lekeyi izle, fotoğrafını arada bir çek ve bir hekime danışmayı düşün."],
  yuksek: ["Hekime görünmeni öneririz", "Birden fazla uyarı işareti var. En kısa sürede bir dermatoloji uzmanına başvur."]
};
const SEV_AD = { dusuk: "Düşük", orta: "Orta", yuksek: "Yüksek" };
const kelime = (v) => (v >= 60 ? "Yüksek" : v >= 30 ? "Orta" : "Düşük");

function cubukEkle(ul, ad, yazi, yuzde) {
  const li = document.createElement("li");
  const satir = document.createElement("span");
  const a = document.createElement("span");
  a.textContent = ad;
  const y = document.createElement("span");
  y.textContent = yazi;
  satir.append(a, y);
  const cubuk = document.createElement("b");
  const dolgu = document.createElement("i");
  dolgu.style.width = Math.max(2, yuzde) + "%";
  cubuk.append(dolgu);
  li.append(satir, cubuk);
  ul.append(li);
}

/* ---------- Analiz ---------- */
async function analiz() {
  const hata = $("#hata");
  hata.hidden = true;
  const isaretli = [...document.querySelectorAll(".belirti:checked")];
  const sayi = isaretli.length;
  const fotoVar = !$("#onizleme").hidden;
  if (!fotoVar && sayi === 0) {
    hata.textContent = "Önce bir fotoğraf yükle ya da en az bir belirti işaretle.";
    hata.hidden = false;
    return;
  }
  const dugme = $("#analiz");
  dugme.disabled = true;
  dugme.textContent = "İnceleniyor...";
  let p = null, oto = null;
  if (fotoVar) {
    try { oto = await fotoAnalizEt(); } catch (e) { console.error(e); }
    if (model) { try { p = await tahminEt(); } catch (e) { console.error(e); } }
  }
  dugme.disabled = false;
  dugme.textContent = "Ön değerlendirmeyi göster";

  const dikkat = p ? DIKKAT.reduce((t, k) => t + p[SINIFLAR.indexOf(k)], 0) : 0;
  const skorlar = oto ? [oto.asim, oto.kenar, oto.renk] : [];
  const yuksekSayi = skorlar.filter((v) => v >= 60).length;
  const efektif = sayi + Math.floor(yuksekSayi / 2);
  const seviye = seviyeHesapla(efektif, dikkat);

  const kart = $("#sonuc");
  kart.className = "kart sonuc " + (seviye === "dusuk" ? "" : seviye);
  $("#rozet").textContent = SEV_AD[seviye];
  $("#s-baslik").textContent = METIN[seviye][0];
  $("#s-metin").textContent = METIN[seviye][1];
  const puan = Math.min(1, (efektif / 7) * 0.6 + dikkat * 0.6);
  $("#s-olcu").style.width = Math.max(6, Math.round(puan * 100)) + "%";

  /* Otomatik ABCD */
  const otoKutu = $("#s-oto"), otoListe = $("#oto-liste");
  otoListe.replaceChildren();
  otoKutu.hidden = !fotoVar;
  $("#oto-tuval").hidden = !oto;
  if (fotoVar) {
    if (oto) {
      $("#oto-not").textContent = "Mavi çizgi, otomatik olarak bulunan leke sınırıdır. Bu puanlar kaba bir tahmindir, ölçüm değildir.";
      cubukEkle(otoListe, "Asimetri", kelime(oto.asim) + " · " + oto.asim, oto.asim);
      cubukEkle(otoListe, "Kenar düzensizliği", kelime(oto.kenar) + " · " + oto.kenar, oto.kenar);
      cubukEkle(otoListe, "Renk çeşitliliği", kelime(oto.renk) + " · " + oto.renk, oto.renk);
    } else {
      $("#oto-not").textContent = "Leke net seçilemedi. Lekeyi fotoğrafın ortasına alıp, sade bir zeminde, iyi ışıkta ve yakından çekmeyi dene.";
    }
  }

  /* Model tahmini */
  const liste = $("#s-liste");
  liste.replaceChildren();
  $("#s-model").hidden = !p;
  const top3 = p ? p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3) : [];
  top3.forEach(([v, i]) => cubukEkle(liste, ADLAR[SINIFLAR[i]], "%" + Math.round(v * 100), Math.round(v * 100)));

  /* Takip: aynı bölgedeki önceki ölçümle karşılaştır */
  const bolge = $("#bolge").value;
  const takip = $("#s-takip");
  takip.hidden = true;
  const onceki = gecmisOku().find((k) => k.b === bolge && k.a);
  if (oto && onceki) {
    const f = [oto.asim - onceki.a[0], oto.kenar - onceki.a[1], oto.renk - onceki.a[2]];
    const isaret = (v) => (v > 0 ? "+" : "") + v;
    takip.textContent = "Önceki " + bolge.toLowerCase() + " ölçümüne göre (" + onceki.t + "): asimetri " + isaret(f[0]) +
      ", kenar " + isaret(f[1]) + ", renk " + isaret(f[2]) + " puan." +
      (f.some((v) => Math.abs(v) >= 15) ? " Belirgin bir değişim var, bunu bir hekime göster." : " Büyük bir değişim görünmüyor.");
    takip.hidden = false;
  }

  let ozet = sayi + " / 7 belirti işaretlendi. Bölge: " + bolge + ". Süre: " + $("#sure").value + ".";
  if (fotoVar && !p) ozet += " Yapay zeka modeli kullanılamadı.";
  $("#s-ozet").textContent = ozet;

  /* Kopyalanacak özet */
  sonOzet = ["Berrak ön değerlendirme özeti (teşhis değildir)",
    "Tarih: " + new Date().toLocaleDateString("tr-TR"),
    "Bölge: " + bolge + " | Süre: " + $("#sure").value,
    "Seviye: " + SEV_AD[seviye],
    "Belirtiler: " + (sayi ? isaretli.map((x) => x.nextElementSibling.textContent).join("; ") : "işaretlenmedi"),
    oto ? "Otomatik ABCD (0-100): asimetri " + oto.asim + ", kenar " + oto.kenar + ", renk " + oto.renk : "",
    top3.length ? "Model tahmini: " + top3.map(([v, i]) => ADLAR[SINIFLAR[i]] + " %" + Math.round(v * 100)).join(" | ") : "",
    "Bu özet bir ders projesinden gelir. Lütfen bir dermatoloji uzmanına görünün."].filter(Boolean).join("\n");

  kart.hidden = false;
  kart.scrollIntoView({ behavior: "smooth", block: "start" });
  kart.focus({ preventScroll: true });
  gecmiseEkle({ t: new Date().toLocaleDateString("tr-TR"), b: bolge, n: sayi, s: seviye, a: oto ? [oto.asim, oto.kenar, oto.renk] : null });
}
$("#analiz").addEventListener("click", analiz);
$("#yazdir").addEventListener("click", () => window.print());

$("#kopyala").addEventListener("click", async () => {
  const d = $("#kopya-durum");
  try {
    await navigator.clipboard.writeText(sonOzet);
    d.textContent = "Özet kopyalandı. Hekimine gösterebilir ya da mesajla gönderebilirsin.";
  } catch (e) {
    d.textContent = "Kopyalanamadı. Metni elle seçip kopyalayabilirsin.";
  }
  d.hidden = false;
});

/* Aylık kontrol için takvim hatırlatıcısı (.ics) */
$("#hatirlat").addEventListener("click", () => {
  const z = (n) => String(n).padStart(2, "0");
  const al = (dt) => dt.getFullYear() + z(dt.getMonth() + 1) + z(dt.getDate());
  const ilk = new Date();
  ilk.setMonth(ilk.getMonth() + 1);
  const ics = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Berrak//TR", "BEGIN:VEVENT",
    "UID:berrak-" + Date.now() + "@berrak", "DTSTAMP:" + al(new Date()) + "T090000Z",
    "DTSTART;VALUE=DATE:" + al(ilk), "RRULE:FREQ=MONTHLY",
    "SUMMARY:Aylık cilt kontrolü", "DESCRIPTION:Lekelerini fotoğrafla ve değişiklik var mı bak. Şüpheli değişiklikte hekime başvur.",
    "BEGIN:VALARM", "TRIGGER:PT9H", "ACTION:DISPLAY", "DESCRIPTION:Aylık cilt kontrolü", "END:VALARM",
    "END:VEVENT", "END:VCALENDAR"].join("\r\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([ics], { type: "text/calendar" }));
  a.download = "aylik-cilt-kontrolu.ics";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
});

/* ---------- Geçmiş (yalnızca bu cihazda, fotoğraf kaydedilmez) ---------- */
const ANAHTAR = "berrak-gecmis";
function gecmisOku() {
  try { return JSON.parse(localStorage.getItem(ANAHTAR)) || []; } catch (e) { return []; }
}
function gecmiseEkle(k) {
  try {
    const g = gecmisOku();
    g.unshift(k);
    localStorage.setItem(ANAHTAR, JSON.stringify(g.slice(0, 10)));
  } catch (e) {}
  gecmisCiz();
}
function gecmisCiz() {
  const ul = $("#gecmis-liste");
  ul.replaceChildren();
  const g = gecmisOku();
  $("#gecmis-sil").hidden = g.length === 0;
  if (!g.length) {
    const li = document.createElement("li");
    li.textContent = "Henüz bir değerlendirme yok.";
    ul.append(li);
    return;
  }
  g.forEach((k) => {
    const li = document.createElement("li");
    const sol = document.createElement("span");
    sol.textContent = k.t + " · " + k.b;
    const sag = document.createElement("span");
    sag.textContent = k.n + " belirti · " + SEV_AD[k.s] + (k.a ? " · ABCD " + k.a.join("/") : "");
    li.append(sol, sag);
    ul.append(li);
  });
}
$("#gecmis-sil").addEventListener("click", () => {
  try { localStorage.removeItem(ANAHTAR); } catch (e) {}
  gecmisCiz();
});

/* ---------- Cilt tipi notları ---------- */
const CILT = {
  "": "Bir cilt tipi seç, sana kısa bir not göstereyim.",
  "1": "Çok açık ten: Cilt kanseri riski en yüksek gruptur. SPF 50 ve üzeri krem, gölge ve kapalı giysiyi hiç aksatma, düzenli cilt muayenesi yaptır.",
  "2": "Açık ten: Risk yüksektir. SPF 50 krem, şapka ve gözlük kullan, öğle güneşinden kaçın.",
  "3": "Orta ten: SPF 30 ya da üzeri krem kullan ve 2 saatte bir yenile. Yanıklardan özellikle kaçın.",
  "4": "Buğday ya da zeytin ten: Az yansan da UV hasarı birikir. SPF 30 ya da üzeri kullanmayı sürdür.",
  "5": "Koyu ten: Güneşe bağlı risk daha düşüktür ama sıfır değildir ve kanser daha geç fark edilebilir. SPF 30 kullan, avuç içi, ayak tabanı ve tırnak altını da kontrol et.",
  "6": "Çok koyu ten: Hemen hiç yanmasan da cilt kanseri olabilir ve geç fark edilebilir. SPF 30 kullan, avuç içi, ayak tabanı ve tırnak altını düzenli kontrol et."
};
$("#cilt-tipi").addEventListener("change", (e) => {
  $("#cilt-tipi-sonuc").textContent = CILT[e.target.value] || CILT[""];
});

gecmisCiz();
modeliYukle();
