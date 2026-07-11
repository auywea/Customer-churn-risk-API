create table clients (
    client_id varchar(20) primary key,
    gender smallint,
    senior_citizen smallint, 
    partner smallint,
    dependents smallint,
    tenure_months integer,
    contract varchar(20),
    payment_method varchar(30),
    monthly_charges numeric(8,2),
    created_at timestamptz not null default now()
);

create table predictions (
    id bigserial primary key,
    client_id varchar(20) not null references clients(client_id),
    features jsonb not null,
    probability numeric(5,4) not null check (probability between 0 and 1),
    model_version varchar(20) not null,
    predicted_at timestamptz not null default now()
);

create index idx_predictions_client_id on predictions(client_id);
create index idx_predictions_predicted_at on predictions(predicted_at);