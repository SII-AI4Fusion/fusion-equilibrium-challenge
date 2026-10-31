//! Opt-in, bounded geometry-only caches. Exact byte keys avoid hash collisions.
use data_tree::{DataTree, DataValue};
use ndarray::Array1;
use std::sync::{Mutex, atomic::{AtomicU64, Ordering}};

static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
pub fn stats() -> (u64, u64) { (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed)) }
pub struct GeometryCache(Mutex<Option<(Vec<u8>, DataValue)>>);
impl GeometryCache {
    pub const fn new() -> Self { Self(Mutex::new(None)) }
    fn enabled() -> bool { std::env::var("GSFIT_GEOMETRY_CACHE").as_deref() == Ok("1") }
    pub fn get(&self, key: &[u8]) -> Option<DataValue> {
        if !Self::enabled() { return None; }
        let guard = self.0.lock().unwrap();
        if let Some((stored, value)) = guard.as_ref() {
            if stored == key { HITS.fetch_add(1, Ordering::Relaxed); return Some(value.clone()); }
        }
        MISSES.fetch_add(1, Ordering::Relaxed); None
    }
    pub fn put(&self, key: Vec<u8>, value: DataValue) {
        if Self::enabled() { *self.0.lock().unwrap() = Some((key, value)); }
    }
}
pub fn scalar(key: &mut Vec<u8>, value: f64) { key.extend_from_slice(&value.to_bits().to_le_bytes()); }
pub fn array(key: &mut Vec<u8>, value: &Array1<f64>) {
    key.extend_from_slice(&(value.len() as u64).to_le_bytes());
    for x in value { scalar(key, *x); }
}
fn string(key: &mut Vec<u8>, value: &str) {
    key.extend_from_slice(&(value.len() as u64).to_le_bytes()); key.extend_from_slice(value.as_bytes());
}
// Geometry contains only these variants. Fail closed if upstream extends its schema.
fn encode(key: &mut Vec<u8>, value: &DataValue) {
    match value {
        DataValue::F64(x) => { key.push(0); scalar(key, *x); },
        DataValue::Usize(x) => { key.push(1); key.extend_from_slice(&(*x as u64).to_le_bytes()); },
        DataValue::String(x) => { key.push(2); string(key, x); },
        DataValue::Bool(x) => { key.push(3); key.push(*x as u8); },
        DataValue::Array1(x) => { key.push(4); array(key, x); },
        DataValue::DataTree(tree) => {
            key.push(5); let names = tree.keys(); key.extend_from_slice(&(names.len() as u64).to_le_bytes());
            for name in names { string(key, &name); encode(key, &tree.data[&name]); }
        },
        DataValue::Array2(x) => { key.push(6); for d in x.shape() { key.extend_from_slice(&(*d as u64).to_le_bytes()); } for v in x { scalar(key,*v); } },
        DataValue::Array3(x) => { key.push(7); for d in x.shape() { key.extend_from_slice(&(*d as u64).to_le_bytes()); } for v in x { scalar(key,*v); } },
        DataValue::VecUsize(x) => { key.push(8); key.extend_from_slice(&(x.len() as u64).to_le_bytes()); for v in x { key.extend_from_slice(&(*v as u64).to_le_bytes()); } },
        DataValue::VecBool(x) => { key.push(9); key.extend_from_slice(&(x.len() as u64).to_le_bytes()); for v in x { key.push(*v as u8); } },
    }
}
pub fn subtree(key: &mut Vec<u8>, tree: &DataTree, name: &str) {
    string(key, name);
    match tree.data.get(name) { Some(value) => { key.push(1); encode(key, value); }, None => key.push(0) }
}
pub fn sensors_key(key: &mut Vec<u8>, tree: &DataTree) {
    let names=tree.keys(); key.extend_from_slice(&(names.len() as u64).to_le_bytes());
    for name in names {
        string(key, &name);
        if let DataValue::DataTree(sensor) = &tree.data[&name] { subtree(key, sensor, "geometry"); }
        else { panic!("invalid sensor geometry"); }
    }
}
pub fn coils_key(key: &mut Vec<u8>, coils: &DataTree) {
    match coils.data.get("pf") {
        Some(DataValue::DataTree(pf)) => sensors_key(key, pf),
        None => key.extend_from_slice(&0_u64.to_le_bytes()),
        _ => panic!("invalid PF geometry"),
    }
}
pub fn sensor_greens(tree: &DataTree, fields: &[&str]) -> DataValue {
    let mut saved=DataTree::new();
    for name in tree.keys() {
        let DataValue::DataTree(sensor)=&tree.data[&name] else { panic!("invalid sensor"); };
        if let Some(DataValue::DataTree(greens))=sensor.data.get("greens") {
            for field in fields { if let Some(value)=greens.data.get(*field) {
                saved.get_or_insert(&name).data.insert((*field).to_string(), value.clone());
            }}
        }
    }
    DataValue::DataTree(saved)
}
pub fn restore_sensors(tree: &mut DataTree, value: DataValue) {
    let DataValue::DataTree(saved)=value else { panic!("invalid cached sensors"); };
    for (name, values) in saved.data {
        let DataValue::DataTree(fields)=values else { panic!("invalid cached fields"); };
        tree.get_or_insert(&name).get_or_insert("greens").data.extend(fields.data);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_key_excludes_measurements_but_includes_names_positions_and_angles() {
        let mut a=DataTree::new();
        a.get_or_insert("a").get_or_insert("geometry").insert("r",1.0);
        a.get_or_insert("a").get_or_insert("geometry").insert("angle_pol",0.0);
        let key=|tree:&DataTree| { let mut key=Vec::new(); sensors_key(&mut key,tree); key };
        let before=key(&a);
        a.get_or_insert("a").get_or_insert("b").insert("measured",2.0);
        assert_eq!(before,key(&a));
        a.get_or_insert("a").get_or_insert("geometry").insert("r",1.1);
        assert_ne!(before,key(&a));
        a.get_or_insert("a").get_or_insert("geometry").insert("r",1.0);
        a.get_or_insert("a").get_or_insert("geometry").insert("angle_pol",0.1);
        assert_ne!(before,key(&a));
    }
    #[test]
    fn restore_preserves_fresh_measurements_and_other_green_tables() {
        let mut old=DataTree::new(); old.get_or_insert("a").get_or_insert("greens").insert("plasma",1.0);
        old.get_or_insert("a").get_or_insert("greens").insert("pf",9.0);
        let saved=sensor_greens(&old,&["plasma"]);
        let mut fresh=DataTree::new(); fresh.get_or_insert("a").get_or_insert("b").insert("experimental",2.0);
        fresh.get_or_insert("a").get_or_insert("greens").insert("pf",3.0);
        restore_sensors(&mut fresh,saved);
        assert_eq!(fresh.get("a").get("b").get("experimental").unwrap_f64(),2.0);
        assert_eq!(fresh.get("a").get("greens").get("pf").unwrap_f64(),3.0);
        assert_eq!(fresh.get("a").get("greens").get("plasma").unwrap_f64(),1.0);
    }
}
