    // clang-format off
  private:
    real_matrix stagewise_storage = [this] {
        auto [N, nx, nu, ny, ny_N] = dim;
        return real_matrix{{
            .depth = N + 1,
            .rows = nx*(3*nx) + nx*(nu + nx) + ny*(nu + nx) + 2*(nu + nx)*(nu + 2*nx),
            .cols = 1,
        }};
    }();


  public:
    const mut_real_view HAB = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(0, (nu + nx)*(nu + 2*nx)).reshaped(nu + 2*nx, nu + nx);
    }();

  public:
    const mut_real_view CD = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(3*nu*nx + nu*nu + 2*(nx*nx), ny*(nu + nx)).reshaped(ny, nu + nx);
    }();

  public:
    const mut_real_view LHV = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(3*nu*nx + nu*ny + nu*nu + nx*ny + 2*(nx*nx), (nu + nx)*(nu + 2*nx)).reshaped(nu + 2*nx, nu + nx);
    }();

  public:
    const mut_real_view Wᵀ = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(6*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 4*(nx*nx), nx*(nu + nx)).reshaped(nu + nx, nx);
    }();

  public:
    const mut_real_view LΨU = [this]{
        auto [N, nx, nu, ny, ny_N] = dim;
        return stagewise_storage.view.middle_rows(7*nu*nx + nu*ny + 2*(nu*nu) + nx*ny + 5*(nx*nx), nx*(3*nx)).reshaped(3*nx, nx);
    }();

    // clang-format on
