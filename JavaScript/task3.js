var data={
    name:"Teja",
    id:513,
    gender:"Female",
    city:"tvr",
    address:{
        city:"hyd",
        country:{
            state1:["bhopal","gwa","indore","jabal"],
            state2:"ap",
            state3:"mp",
            state4:"up"
        }
    }
};

console.log(data.address.country.state4);
console.log(data.address.country.state1[2]);