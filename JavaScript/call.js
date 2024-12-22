function f1(){
    console.log("f1 is called");
}
function f2(p1){
    p1();
}
f2(f1);
function fn(p){
    console.log(p);
}
fn(function () {});