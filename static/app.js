// static/app.js
async function confirmPair(){
  const found = document.getElementById('found_id').value;
  const lost = document.getElementById('lost_id').value;
  if(!found || !lost) return alert('enter both ids');
  // For prototype, mark lost item as confirmed (server stores admin_confirmed flag)
  const res = await fetch('/admin/confirm', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({report_id: parseInt(lost), val:1})
  });
  const data = await res.json();
  if(data.ok) alert('Confirmed!');
}
