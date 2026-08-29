/* ============================================================
   NutriChat AI — Frontend App Logic
   ============================================================ */

const API = '';   // Same origin (FastAPI serves frontend)
let token = localStorage.getItem('nc_token') || null;
let currentUser = null;
let cart = {};         // { item_id: {item, qty} }
let menuData = [];
let activeFilter = 'All';

// ── Auth ──────────────────────────────────────────────────────

function switchTab(tab) {
  document.getElementById('formLogin').style.display    = tab==='login'    ? '' : 'none';
  document.getElementById('formRegister').style.display = tab==='register' ? '' : 'none';
  document.getElementById('tabLogin').classList.toggle('active',    tab==='login');
  document.getElementById('tabRegister').classList.toggle('active', tab==='register');
  document.getElementById('loginError').textContent = '';
  document.getElementById('regError').textContent   = '';
}

async function doLogin() {
  const u = document.getElementById('loginUser').value.trim();
  const p = document.getElementById('loginPass').value;
  if (!u || !p) { document.getElementById('loginError').textContent='Fill all fields.'; return; }
  const res = await apiPost('/auth/login', {username:u, password:p}, false);
  if (res.error) { document.getElementById('loginError').textContent = res.error || res.detail; return; }
  token = res.token;
  localStorage.setItem('nc_token', token);
  currentUser = res;
  showApp();
}

async function doRegister() {
  const u  = document.getElementById('regUser').value.trim();
  const e  = document.getElementById('regEmail').value.trim();
  const p  = document.getElementById('regPass').value;
  const g  = parseInt(document.getElementById('regGoal').value) || 2000;
  if (!u||!e||!p) { document.getElementById('regError').textContent='Fill all fields.'; return; }
  const res = await apiPost('/auth/register', {username:u, email:e, password:p, daily_calorie_goal:g}, false);
  if (res.error||res.detail) { document.getElementById('regError').textContent = res.error||res.detail; return; }
  showToast('Account created! Please login 🎉');
  switchTab('login');
}

async function doLogout() {
  await apiPost('/auth/logout', {});
  token = null;
  localStorage.removeItem('nc_token');
  currentUser = null;
  location.reload();
}

async function showApp() {
  document.getElementById('authModal').style.display = 'none';
  document.getElementById('mainApp').style.display   = 'flex';
  document.getElementById('sidebarUsername').textContent = currentUser.username || currentUser.display_name || 'User';
  document.getElementById('sidebarGoal').textContent = `Goal: ${currentUser.daily_calorie_goal||2000} kcal`;
  document.getElementById('userAvatar').textContent  = (currentUser.username||'U')[0].toUpperCase();

  await loadMenu();
  await refreshTracker();
  showWelcomeMessage();
}

// ── API Helpers ───────────────────────────────────────────────

async function apiPost(path, body, auth=true) {
  const headers = {'Content-Type':'application/json'};
  if (auth && token) headers['Authorization'] = `Bearer ${token}`;
  try {
    const r = await fetch(API+path, {method:'POST', headers, body:JSON.stringify(body)});
    const data = await r.json();
    if (!r.ok) return {error: data.detail || JSON.stringify(data)};
    return data;
  } catch(e) { return {error: e.message}; }
}

async function apiGet(path) {
  try {
    const r = await fetch(API+path, {headers: token?{Authorization:`Bearer ${token}`}:{}});
    const data = await r.json();
    if (!r.ok) return {error: data.detail};
    return data;
  } catch(e) { return {error: e.message}; }
}

// ── Chat ──────────────────────────────────────────────────────

function showWelcomeMessage() {
  const name = currentUser?.username || currentUser?.display_name || 'there';
  appendBotMessage(
    `👋 Namaste, **${name}**! I'm NutriChat AI 🌿\n\n` +
    `I can help you with:\n` +
    `• 🔥 Calorie lookup — *'calories in paneer'*\n` +
    `• 🧪 Nutrition info — *'protein in chicken'*\n` +
    `• 🍽️ Meal suggestions — *'suggest healthy lunch'*\n` +
    `• 💊 Diet advice — *'diet tips for weight loss'*\n` +
    `• 📊 Today's intake — *'what have I eaten today'*\n` +
    `• 📏 Portions — *'how much rice should I eat'*\n\n` +
    `What would you like to know? 😊`
  );
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const text  = input.value.trim();
  if (!text) return;
  input.value = '';
  appendUserMessage(text);
  const typingId = showTyping();
  const res = await apiPost('/chat', {message: text});
  removeTyping(typingId);
  if (res.error) {
    appendBotMessage('⚠️ ' + res.error, null);
  } else {
    appendBotMessage(res.response, {
      source: res.ml_source,
      confidence: res.confidence,
      intent: res.intent,
      calorie_estimate: res.calorie_estimate,
    });
    await refreshTracker();
  }
}

function sendChip(text) {
  document.getElementById('chatInput').value = text;
  sendMessage();
}

function appendUserMessage(text) {
  const win = document.getElementById('chatWindow');
  const initial = (currentUser?.username || 'U')[0].toUpperCase();
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `
    <div class="msg-avatar">${initial}</div>
    <div class="msg-bubble">${escHtml(text)}</div>`;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}

function appendBotMessage(markdown, mlMeta) {
  const win = document.getElementById('chatWindow');
  const div = document.createElement('div');
  div.className = 'msg bot';

  let mlBadge = '';
  if (mlMeta && mlMeta.source) {
    const conf = Math.round((mlMeta.confidence || 0) * 100);
    const isML = mlMeta.source === 'ml';
    const color = isML
      ? (conf >= 80 ? '#22c55e' : conf >= 50 ? '#f59e0b' : '#94a3b8')
      : '#6366f1';
    const label = isML ? `🧠 ML ${conf}%` : `⚡ keyword`;
    const intent = (mlMeta.intent || '').replace(/_/g, ' ');
    mlBadge = `<span class="ml-badge" style="border-color:${color};color:${color}" title="Intent: ${intent}">${label}</span>`;
  }

  div.innerHTML = `
    <div class="msg-avatar">🌿</div>
    <div class="msg-bubble">${renderMd(markdown)}${mlBadge}</div>`;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
}

function showTyping() {
  const win = document.getElementById('chatWindow');
  const id  = 'typing_' + Date.now();
  const div = document.createElement('div');
  div.className = 'msg bot'; div.id = id;
  div.innerHTML = `<div class="msg-avatar">🌿</div>
    <div class="msg-bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  win.appendChild(div);
  win.scrollTop = win.scrollHeight;
  return id;
}
function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function renderMd(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/_(.+?)_/g, '<em>$1</em>')
    .replace(/^• (.+)$/gm, '→ $1')
    .replace(/\n/g, '<br>');
}
function escHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Menu ──────────────────────────────────────────────────────

async function loadMenu() {
  const data = await apiGet('/menu');
  if (data.error) return;
  menuData = data;
  renderMenu(data);
}

function filterMenu(type, btn) {
  activeFilter = type;
  document.querySelectorAll('.filter-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  let filtered = menuData;
  if (type === 'Veg')     filtered = menuData.filter(i => i.veg);
  if (type === 'Non-Veg') filtered = menuData.filter(i => !i.veg);
  renderMenu(filtered);
}

function renderMenu(items) {
  const grid = document.getElementById('menuGrid');
  if (!items.length) { grid.innerHTML='<p style="color:var(--text2);text-align:center">No items found.</p>'; return; }

  // Group by category
  const cats = {};
  items.forEach(it => {
    if (!cats[it.category]) cats[it.category] = [];
    cats[it.category].push(it);
  });

  grid.innerHTML = Object.entries(cats).map(([cat, citems]) => `
    <div>
      <div class="cat-label">${cat}</div>
      <div class="menu-items-grid">
        ${citems.map(item => menuCard(item)).join('')}
      </div>
    </div>
  `).join('');
}

function menuCard(item) {
  const inCart = cart[item.id];
  const qty    = inCart ? inCart.qty : 0;
  const tags   = (item.tags||[]).map(t=>`<span class="item-tag">${t}</span>`).join('');
  const vegDot = item.veg ? '🟢' : '🔴';

  const ctrl = qty > 0
    ? `<div class="qty-ctrl">
         <button onclick="updateCart('${item.id}',-1)">−</button>
         <span>${qty}</span>
         <button onclick="updateCart('${item.id}',1)">+</button>
       </div>`
    : `<button class="add-btn" onclick="updateCart('${item.id}',1)">+</button>`;

  return `
    <div class="menu-card" id="card_${item.id}">
      ${ctrl}
      <div class="item-emoji">${item.emoji||'🍽️'} <span style="font-size:12px">${vegDot}</span></div>
      <div class="item-name">${item.name}</div>
      <div class="item-cat">${item.category}</div>
      <div class="item-stats">
        <span class="item-cals">🔥 ${item.calories_kcal} kcal</span>
        <span class="item-price">₹${item.price}</span>
      </div>
      <div class="item-tags" style="margin-top:6px">${tags}</div>
      <div style="font-size:11px;color:var(--text2);margin-top:4px">
        💪${item.protein_g}g  🌾${item.carb_g}g  🧈${item.fat_g}g
      </div>
    </div>`;
}

function updateCart(itemId, delta) {
  const item = menuData.find(i=>i.id===itemId);
  if (!item) return;
  const cur = cart[itemId] ? cart[itemId].qty : 0;
  const nqty = Math.max(0, cur + delta);
  if (nqty === 0) delete cart[itemId];
  else cart[itemId] = {item, qty: nqty};
  updateCartBar();
  // Re-render only the card
  const card = document.getElementById('card_'+itemId);
  if (card) card.outerHTML = menuCard(item);
}

function updateCartBar() {
  const items = Object.values(cart);
  const count = items.reduce((s,i)=>s+i.qty, 0);
  const total = items.reduce((s,i)=>s+i.item.price*i.qty, 0);
  const cals  = items.reduce((s,i)=>s+i.item.calories_kcal*i.qty, 0);
  const bar   = document.getElementById('cartBar');
  document.getElementById('cartCount').textContent = count;
  document.getElementById('cartTotal').textContent = total;
  document.getElementById('cartCals').textContent  = Math.round(cals);
  bar.style.display = count > 0 ? 'flex' : 'none';
}

// ── Cart Modal ────────────────────────────────────────────────

function openCart() {
  const items = Object.values(cart);
  if (!items.length) { showToast('Your cart is empty!'); return; }
  const cartItemsEl = document.getElementById('cartItems');
  cartItemsEl.innerHTML = items.map(({item, qty}) => `
    <div class="cart-item">
      <div class="ci-emoji">${item.emoji||'🍽️'}</div>
      <div class="ci-info">
        <div class="ci-name">${item.name}</div>
        <div class="ci-detail">${item.calories_kcal * qty} kcal · ₹${item.price * qty}</div>
      </div>
      <div class="ci-qty">
        <button onclick="updateCart('${item.id}',-1);openCart()">−</button>
        <span>${qty}</span>
        <button onclick="updateCart('${item.id}',1);openCart()">+</button>
      </div>
      <div class="ci-price">₹${item.price*qty}</div>
    </div>`).join('');

  const total = items.reduce((s,i)=>s+i.item.price*i.qty, 0);
  const cals  = items.reduce((s,i)=>s+i.item.calories_kcal*i.qty, 0);
  const prot  = items.reduce((s,i)=>s+(i.item.protein_g||0)*i.qty, 0);
  document.getElementById('cartSummary').innerHTML = `
    <div class="summary-row"><span>🔥 Total Calories</span><span>${Math.round(cals)} kcal</span></div>
    <div class="summary-row"><span>💪 Protein</span><span>${prot.toFixed(1)}g</span></div>
    <div class="summary-row total"><span>Total Amount</span><span>₹${total}</span></div>`;

  document.getElementById('cartModal').style.display = 'flex';
}

function closeCart() {
  document.getElementById('cartModal').style.display = 'none';
}

async function placeOrder() {
  const items = Object.values(cart);
  if (!items.length) return;
  const orderItems = items.map(({item, qty}) => ({item_id: item.id, qty}));
  const res = await apiPost('/orders', {items: orderItems});
  if (res.error) { showToast('Order failed: '+res.error); return; }
  showToast(`✅ Order #${res.order_id} placed! ${Math.round(res.total_calories)} kcal logged.`);
  cart = {};
  updateCartBar();
  closeCart();
  renderMenu(menuData);
  await refreshTracker();
  appendBotMessage(
    `✅ **Order Placed Successfully!**\n\n` +
    `Order ID: ${res.order_id}\n` +
    `Total: ₹${res.total_price} · ${Math.round(res.total_calories)} kcal\n\n` +
    `All items have been logged to your daily tracker. Would you like me to suggest your next meal? 🍽️`
  );
  showPanel('chat');
}

// ── Orders Panel ──────────────────────────────────────────────

async function loadOrders() {
  const list = document.getElementById('orderList');
  list.innerHTML = '<p style="color:var(--text2);text-align:center;padding:40px">Loading...</p>';
  const data = await apiGet('/orders/history');
  if (data.error||!data.length) {
    list.innerHTML = '<p style="color:var(--text2);text-align:center;padding:40px">No orders yet. Go order something! 🍽️</p>';
    return;
  }
  list.innerHTML = data.map(order => `
    <div class="order-card">
      <div class="order-header">
        <div>
          <div class="order-id">Order #${order.order_id}</div>
          <div class="order-time">${order.time_str}</div>
        </div>
        <span class="order-status">${order.status}</span>
      </div>
      <div class="order-items-list">
        ${(order.items||[]).map(it=>`
          <div class="oi-row">
            <span class="oi-name">${it.emoji||'🍽️'} ${it.name} ×${it.qty}</span>
            <span>₹${it.subtotal}</span>
          </div>`).join('')}
      </div>
      <div class="order-footer">
        <span class="order-total">₹${order.total_price}</span>
        <span class="order-cals">🔥 ${Math.round(order.total_calories)} kcal</span>
      </div>
    </div>`).join('');
}

// ── Tracker ───────────────────────────────────────────────────

async function refreshTracker() {
  const data = await apiGet('/tracker/today');
  if (data.error) return;

  const goal    = data.daily_goal || 2000;
  const cals    = data.total_calories || 0;
  const pct     = Math.min(100, (cals / goal) * 100);
  const circumf = 314;
  const offset  = circumf - (circumf * pct / 100);

  document.getElementById('ringCals').textContent = Math.round(cals);
  document.getElementById('ringGoal').textContent = goal;
  document.getElementById('ringCircle').style.strokeDashoffset = offset;
  document.getElementById('macroP').textContent = (data.protein_g||0).toFixed(1)+'g';
  document.getElementById('macroC').textContent = (data.carb_g||0).toFixed(1)+'g';
  document.getElementById('macroF').textContent = (data.fat_g||0).toFixed(1)+'g';
  document.getElementById('remainingCals').textContent = Math.round(data.remaining_calories||goal) + ' kcal';

  // Meals list
  const meals = (data.meals||[]).slice(-5).reverse();
  const ml    = document.getElementById('mealList');
  if (!meals.length) {
    ml.innerHTML = '<div class="empty-meals">No meals logged yet</div>';
  } else {
    ml.innerHTML = meals.map(m=>`
      <div class="meal-item">
        <div>
          <div class="m-name">${m.food_name}</div>
          <div class="m-time">${m.time_str||''}</div>
        </div>
        <div class="m-cal">🔥${Math.round(m.calories)}</div>
      </div>`).join('');
  }
}

// ── Panels ────────────────────────────────────────────────────

function showPanel(name) {
  ['chat','menu','orders'].forEach(p => {
    const el = document.getElementById('panel'+p.charAt(0).toUpperCase()+p.slice(1));
    if (el) el.style.display = p===name ? 'flex' : 'none';
    const btn = document.getElementById('nav'+p.charAt(0).toUpperCase()+p.slice(1));
    if (btn) btn.classList.toggle('active', p===name);
  });
  if (name === 'orders') loadOrders();
  if (name === 'menu')   renderMenu(menuData);
}

// ── Toast ─────────────────────────────────────────────────────

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'), 3500);
}

// ── Init ──────────────────────────────────────────────────────

(async function init() {
  if (token) {
    const res = await apiGet('/auth/me');
    if (!res.error) {
      currentUser = res;
      await showApp();
      return;
    }
    localStorage.removeItem('nc_token');
    token = null;
  }
  document.getElementById('authModal').style.display = 'flex';
  document.getElementById('mainApp').style.display   = 'none';
})();
