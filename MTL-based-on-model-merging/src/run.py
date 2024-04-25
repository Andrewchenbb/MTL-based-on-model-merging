import Linear_TA_merge
import Linear_TA_minus
import TIES_minus
import ada_TIES
import ada_linearTA
import TA_minus
import TIES_merge

def safe_run(func):
    try:
        func()
    except Exception as e:
        print(f"Error running {func.__name__}: {e}")

if __name__=='__main__':
    print("执行TA_minus")
    TA_minus.main()
    print("执行Ties_minus")
    safe_run(TIES_minus.main())
    #safe_run(TIES_merge.main())
    print("执行linear_ta-merge")
    safe_run(Linear_TA_merge.main())
    print("执行Linear_TA_minus")
    safe_run(Linear_TA_minus.main())

    #ada_TIES.main()
    #ada_linearTA.main()
